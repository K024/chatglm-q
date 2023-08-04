import math
import torch
from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class QwenConfig():
    hidden_size: int = 4096
    inner_hidden_size: int = 11008
    head_hidden_size: int = 128

    num_attention_heads: int = 32
    num_layers: int = 32

    vocab_size: int = 151936
    dropout_rate: float = 0.0
    layernorm_epsilon: float = 1e-05
    max_sequence_length: int = 2048


# not used
def precompute_sinusoids(dim: int, length: int, scale = 10000.0):
    assert dim % 2 == 0
    log_timescale_increment = torch.log(scale) / (dim // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(dim // 2))
    scaled_time = torch.outer(torch.arange(length).float(), inv_timescales)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def precompute_freqs_cis(dim: int, length: int, theta = 10000.0):
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(length).float(), freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(
    x: Tensor,                      # (*, d_head)
    freqs_cis: tuple[Tensor, ...],  # (*, d_head // 2), (*, d_head // 2)
    d_head: int,
) -> Tensor:
    x_r, x_i = torch.split(x, d_head // 2, dim=-1)
    f_r, f_i = freqs_cis
    o_r = x_r * f_r - x_i * f_i
    o_i = x_r * f_i + x_i * f_r
    return torch.cat([o_r, o_i], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: tuple[int, ...], eps=1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x),
                        None if self.bias is None else self.bias.type_as(x))

    def reset_parameters(self):
        pass


class Embedding(nn.Embedding):
    def reset_parameters(self):
        pass


class QwenAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        d_head: int,
        dropout_rate = 0.0,
        qkv_bias = True,
        o_bias = False,
        dtype = None,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        assert n_state % (n_head * 4) == 0
        self.qkv_proj = Linear(n_state, d_head * n_head * 3, bias=qkv_bias, dtype=dtype)
        self.o_proj = Linear(d_head * n_head, n_state, bias=o_bias, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: Tensor,
        freqs_cis: tuple[Tensor, ...],
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, ...]] = None,
    ):
        '''
        x:
            Shape: (n_batch, n_seq or n_seq_new (using cache), n_state)

        freqs_cis:
            Shape: (n_batch, n_seq or n_seq_new, 1, d_head // 2) * 2

        attention_mask:
            0 for no mask, -inf for masked
            Shape: (n_batch, n_seq_new, n_seq)

        kv_cache:
            Tuple of (k_cache, v_cache)
        '''
        n_batch, n_seq, _ = x.shape
        d_head, n_head = self.d_head, self.n_head

        q, k, v = torch.split(self.qkv_proj(x), d_head * n_head, dim=-1)

        q = q.view(n_batch, n_seq, n_head, d_head)
        k = k.view(n_batch, n_seq, n_head, d_head)
        v = v.view(n_batch, n_seq, n_head, d_head)

        q = apply_rotary_emb(q, freqs_cis, d_head)
        k = apply_rotary_emb(k, freqs_cis, d_head)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        kv_cache = (k.detach(), v.detach())

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        # (n_batch, n_head, n_seq, n_seq_past)
        qk = torch.matmul(q, k) / math.sqrt(d_head)
        if attention_mask is not None:
            qk = qk + attention_mask[:, None, :, :]

        scores = F.softmax(qk.float(), dim=-1).type_as(x)
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        output = output.permute(0, 2, 1, 3).reshape(n_batch, n_seq, -1)
        output = self.o_proj(output)

        return output, kv_cache


class GatedFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout_rate = 0.0,
        bias = False,
        dtype = None,
        act_fn = F.silu,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w_in = Linear(dim, hidden_dim, bias=bias, dtype=dtype)
        self.w_gate = Linear(dim, hidden_dim, bias=bias, dtype=dtype)
        self.w_out = Linear(hidden_dim, dim, bias=bias, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_fn = act_fn

    def forward(self, x: Tensor):
        h = self.act_fn(self.w_gate(x)) * self.w_in(x)
        return self.w_out(self.dropout(h))


class QwenBlock(nn.Module):
    def __init__(self, config: QwenConfig, dtype=None):
        super().__init__()
        self.attn_ln = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=dtype)
        self.attn = QwenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.head_hidden_size,
            dropout_rate=config.dropout_rate,
            dtype=dtype)
        self.ffn_ln = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=dtype)
        self.ffn = GatedFeedForward(
            config.hidden_size,
            config.inner_hidden_size,
            config.dropout_rate,
            dtype=dtype)

    def forward(
        self,
        x: Tensor,
        freqs_cis: tuple[Tensor, ...],
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, ...]] = None,
    ):
        h, kv_cache = self.attn(
            x=self.attn_ln(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        x = x + h
        h = self.ffn(self.ffn_ln(x))
        output = x + h
        return output, kv_cache


class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig, dtype=None):
        super().__init__()
        self.config = config
        self.word_embedding = Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, dtype=dtype
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = nn.ModuleList([
            QwenBlock(config, dtype=dtype) for i in range(config.num_layers)
        ])
        self.final_ln = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

        freqs_cis_r, freqs_cis_i = precompute_freqs_cis(
            config.head_hidden_size, config.max_sequence_length)
        self.register_buffer("freqs_cis_r", freqs_cis_r.to(dtype=dtype), persistent=False)
        self.register_buffer("freqs_cis_i", freqs_cis_i.to(dtype=dtype), persistent=False)

    def prepare_input(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[Tensor, ...], ...]] = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        """
        returns: (
            input_embeddings,
            attention_mask,
            freqs_cis,
        )
        """
        if input_embeddings is None:
            assert input_ids is not None, "No input"
            device = input_ids.device
            input_embeddings = self.word_embedding(input_ids)
            n_batch, n_seq_new = input_ids.shape
        else:
            assert input_ids is None, "Specify either 'input_ids' or 'input_embeddings'"
            device = input_embeddings.device
            n_batch, n_seq_new, _ = input_embeddings.shape

        if past_key_values is not None:
            n_seq_past = past_key_values[0][0].shape[1]
            n_seq = n_seq_new + n_seq_past
        else:
            n_seq = n_seq_new

        if attention_mask is None:
            attention_mask = torch.ones(n_batch, n_seq, dtype=torch.long, device=device)

        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)

        # causal mask with full prefix attention
        # trilu is not supported in onnxruntime
        seq = torch.arange(n_seq, device=device)
        causal_mask = (seq[:, None] < seq[None, :])
        # make attention_mask to a float causal mask
        attention_mask = (causal_mask[None, ...] | ~attention_mask[:, None, :].bool()).float() * -1e10

        # align to input_ids
        attention_mask = attention_mask[:, -n_seq_new:]
        position_ids = position_ids[:, -n_seq_new:]

        freqs_cis = tuple(
            F.embedding(position_ids, pe).view(n_batch, n_seq_new, 1, -1)
            for pe in [self.freqs_cis_r, self.freqs_cis_i]
        )

        return (
            input_embeddings,
            attention_mask,
            freqs_cis,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[Tensor, ...], ...]] = None,
    ):
        '''
        input_ids:
            Shape: (n_batch, n_seq or n_new)

        attention_mask:
            Shape: (n_batch, n_seq) with 1 for token and 0 for pad

        position_ids:
            Shape: (n_batch, n_seq or n_new) same as input_ids

        labels:
            Same as input_ids (no shift required) with -100 for prefix and pad tokens

        past_key_values:
            Tuple[Tuple[Tensor, Tensor], ...] where each:
            Shape: (n_batch, n_past, num_multi_query_groups, 1, head_hidden_size)
                    n_seq = n_past + n_new
        '''
        (
            input_embeddings,
            attention_mask,
            freqs_cis,
        ) = self.prepare_input(
            input_ids,
            input_embeddings,
            attention_mask,
            position_ids,
            past_key_values,
        )

        # forward layers
        h = self.dropout(input_embeddings)
        current_key_values = tuple()
        for i, layer in enumerate(self.layers):
            kv_cache = past_key_values[i] if past_key_values is not None else None
            h, kv_cache = layer(
                h,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                kv_cache=kv_cache,
            )
            current_key_values += (kv_cache, )

        h = self.final_ln(h)
        output: Tensor = self.lm_head(h)

        if labels is not None:
            n_classes = self.config.vocab_size
            shift_logits = output[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, n_classes), shift_labels.view(-1))
        else:
            loss = None

        return loss, output, current_key_values
