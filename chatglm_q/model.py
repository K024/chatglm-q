import math
import torch
from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ChatGLM2Config():
    hidden_size: int = 4096
    inner_hidden_size: int = 13696
    head_hidden_size: int = 128

    num_multi_query_groups: int = 2
    num_attention_heads: int = 32
    num_layers: int = 28

    vocab_size: int = 65024
    dropout_rate: float = 0.0
    layernorm_epsilon: float = 1e-05
    max_sequence_length: int = 8192


# not used
def precompute_sinusoids(dim: int, length: int, scale = 10000.0):
    assert dim % 2 == 0
    log_timescale_increment = torch.log(scale) / (dim // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(dim // 2))
    scaled_time = torch.outer(torch.arange(length).float(), inv_timescales)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# changed from v1: [r, r, ..., i, i, ...] => [[r, i], [r, i], ...]
def precompute_freqs_cis(dim: int, length: int, theta = 10000.0):
    assert dim % 4 == 0
    # half of the head_dim bypassed
    dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(length).float(), freqs)
    freqs_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    freqs_bypass = torch.stack([torch.ones_like(freqs), torch.zeros_like(freqs)], dim=-1)
    return torch.cat([freqs_cis, freqs_bypass], dim=-2)


# changed from v1
ROTARY_VIEW_AS_COMPLEX = True
def apply_rotary_emb(
    x: Tensor,          # (n_batch, n_seq, n_groups, n_head, d_head)
    freqs_cis: Tensor,  # (n_batch, n_seq, 1, 1, d_head // 2, 2)
    shape: list[int],
) -> Tensor:
    *shape_prefix, d_head = shape
    x = x.view(*shape_prefix, d_head // 2, 2)
    if ROTARY_VIEW_AS_COMPLEX:
        x = torch.view_as_complex(x)
        freqs_cis = torch.view_as_complex(freqs_cis)
        return torch.view_as_real(x * freqs_cis).view(*shape)
    else:
        o_r = x[..., 0] * freqs_cis[..., 0] - x[..., 1] * freqs_cis[..., 1]
        o_i = x[..., 0] * freqs_cis[..., 1] + x[..., 1] * freqs_cis[..., 0]
        return torch.stack([o_r, o_i], dim=-1).view(*shape)


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


class ChatGLM2Attention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        d_head: int,
        n_groups: int,
        layer_idx: int,
        dropout_rate = 0.0,
        qkv_bias = True,
        o_bias = False,
        dtype = None,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.n_groups = n_groups
        assert n_state % (n_head * 4) == 0
        assert n_head % n_groups == 0
        self.layer_idx = layer_idx
        # multi-query attention
        self.qkv_proj = Linear(n_state, d_head * (n_head + 2 * n_groups), bias=qkv_bias, dtype=dtype)
        self.o_proj = Linear(d_head * n_head, n_state, bias=o_bias, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, ...]] = None,
        use_past = False,
    ):
        '''
        x:
            Shape: (n_batch, n_seq or 1 (using cache), n_state)

        freqs_cis:
            Shape: (n_batch, n_seq or 1, 1, 1, d_head // 4, 2)

        attention_mask:
            0 for no mask, -inf for masked
            Shape: (n_batch, n_seq, n_seq)

        kv_cache:
            Tuple of (k_cache, v_cache)
            Shape: (n_batch, n_seq - 1, n_head, d_head)
        '''
        n_batch, n_seq, _ = x.shape
        d_head, n_head, n_groups = self.d_head, self.n_head, self.n_groups

        fused_qkv = self.qkv_proj(x)
        split_size = [d_head * n_head, d_head * n_groups, d_head * n_groups]
        q, k, v = torch.split(fused_qkv, split_size, dim=-1)

        # allow broadcast along groups
        shape_q = [n_batch, n_seq, n_groups, n_head // n_groups, d_head]
        shape_kv = [n_batch, n_seq, n_groups, 1, d_head]
        q = q.view(*shape_q)
        k = k.view(*shape_kv)
        v = v.view(*shape_kv)

        q = apply_rotary_emb(q, freqs_cis, shape_q)
        k = apply_rotary_emb(k, freqs_cis, shape_kv)

        if use_past:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        kv_cache = (k.detach(), v.detach())

        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 4, 1)
        v = v.permute(0, 2, 3, 1, 4)

        # maybe useless, test needed
        scaling_coeff = float(self.layer_idx + 1)
        q = q / (math.sqrt(d_head) * scaling_coeff)

        # (n_batch, n_group, n_heads, n_new, n_seq)
        qk = torch.matmul(q, k) # / math.sqrt(d_head) # no need to scale again
        if attention_mask is not None:
            qk = qk + attention_mask[:, None, None, :, :]

        scores = F.softmax(qk.float() * scaling_coeff, dim=-1).type_as(x)
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        output = output.permute(0, 3, 1, 2, 4).reshape(n_batch, n_seq, -1)
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
        self.hidden_dim = hidden_dim
        # fused gate act
        self.w_in = Linear(dim, hidden_dim * 2, bias=bias, dtype=dtype)
        self.w_out = Linear(hidden_dim, dim, bias=bias, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_fn = act_fn

    def forward(self, x: Tensor):
        h, gate = torch.split(self.w_in(x), self.hidden_dim, dim=-1)
        return self.w_out(self.dropout(self.act_fn(h) * gate))


class ChatGLM2Block(nn.Module):
    def __init__(self, layer_idx: int, config: ChatGLM2Config, dtype=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_ln = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=dtype)
        self.attn = ChatGLM2Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.head_hidden_size,
            config.num_multi_query_groups,
            layer_idx,
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
        freqs_cis: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, ...]] = None,
        use_past = False,
    ):
        h, kv_cache = self.attn(
            x=self.attn_ln(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_past=use_past,
        )
        x = x + h
        h = self.ffn(self.ffn_ln(x))
        output = x + h
        return output, kv_cache


class ChatGLM2Model(nn.Module):
    def __init__(self, config: ChatGLM2Config, dtype=None):
        super().__init__()
        self.config = config
        self.word_embedding = Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, dtype=dtype
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = nn.ModuleList([
            ChatGLM2Block(layer_idx, config, dtype=dtype) for layer_idx in range(config.num_layers)
        ])
        self.final_ln = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

        # half of head_dim bypassed
        d_freqs_cis = config.head_hidden_size
        self.d_freqs_cis = d_freqs_cis
        freqs_cis_cache = precompute_freqs_cis(d_freqs_cis, config.max_sequence_length) \
            .view(config.max_sequence_length, -1).to(dtype=dtype)
        self.register_buffer("freqs_cis_cache", freqs_cis_cache, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[Tensor, ...], ...]] = None,
        use_past = True,
        predict_last_one_token = False,
    ):
        '''
        x:
            Shape: (n_batch, n_seq)

        attention_mask:
            Shape: (n_batch, n_seq) with 1 for token and 0 for pad

        labels:
            Same as x (no shift required) with -100 for prefix and pad tokens
        '''
        if input_embeddings is None:
            assert input_ids is not None, "No input"
            device = input_ids.device
            n_batch, n_seq = input_ids.shape
        else:
            assert input_ids is None, "Specify either 'input_ids' or 'input_embeddings'"
            n_batch, n_seq, _ = input_embeddings.shape
            device = input_embeddings.device

        has_past_key_values = past_key_values is not None
        assert not has_past_key_values or len(past_key_values) > 0, "'past_key_values' is not valid"

        if attention_mask is None:
            attention_mask = torch.ones(n_batch, n_seq, dtype=torch.long, device=device)

        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1) - 1

        # causal mask with full prefix attention
        # trilu is not supported in onnxruntime
        seq = torch.arange(n_seq, device=device)
        causal_mask = (seq[:, None] < seq[None, :])
        # make attention_mask to float causal mask
        attention_mask = (causal_mask[None, ...] | ~attention_mask[:, None, :].bool()).float() * -1e10

        # for incremental generation
        if has_past_key_values:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            else:
                input_embeddings = input_embeddings[:, -1:]
            attention_mask = attention_mask[:, -1:]
            position_ids = position_ids[:, -1:]

        if input_embeddings is None:
            input_embeddings = self.word_embedding(input_ids)

        freqs_cis = F.embedding(position_ids, self.freqs_cis_cache) \
            .view(n_batch, -1, 1, 1, self.d_freqs_cis // 2, 2)

        # forward layers
        h = self.dropout(input_embeddings)
        current_key_values = tuple() if use_past else None
        for i, layer in enumerate(self.layers):
            kv_cache = past_key_values[i] if has_past_key_values else None
            h, kv_cache = layer(
                h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                use_past=has_past_key_values and use_past
            )
            if use_past:
                current_key_values += (kv_cache, )

        h = self.final_ln(h)

        if predict_last_one_token:
            output = self.lm_head(h[:, -1:, :])
            return None, output, current_key_values

        output: Tensor = self.lm_head(h)

        if labels is not None:
            assert not has_past_key_values, "'past_key_values' should not be used when training"
            n_classes = self.config.vocab_size
            shift_logits = output[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, n_classes), shift_labels.view(-1))
        else:
            loss = None

        return loss, output, current_key_values
