import math
import torch
from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ChatGLMConfig():
    hidden_size = 4096
    inner_hidden_size = 16384
    layernorm_epsilon = 1e-05
    max_sequence_length = 2048
    num_attention_heads = 32
    num_layers = 28
    vocab_size = 130528
    position_encoding_2d = True
    dropout_rate = 0.0


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
    freqs_cis_r = torch.cos(freqs)
    freqs_cis_i = torch.sin(freqs)
    return freqs_cis_r, freqs_cis_i


def apply_rotary_emb(
    x_r: Tensor,          # (n_batch, n_seq, n_head, d_head // 2)
    x_i: Tensor,          # (n_batch, n_seq, n_head, d_head // 2)
    freqs_cis_r: Tensor,  # (n_batch, n_seq, d_head // 2)
    freqs_cis_i: Tensor,  # (n_batch, n_seq, d_head // 2)
) -> Tensor:
    freqs_cis_r = freqs_cis_r[:, :, None, :]
    freqs_cis_i = freqs_cis_i[:, :, None, :]
    o_r = x_r * freqs_cis_r - x_i * freqs_cis_i
    o_i = x_r * freqs_cis_i + x_i * freqs_cis_r
    return torch.cat([o_r, o_i], dim=-1)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        weight, bias = self.weight.float(), self.bias.float()
        return F.layer_norm(x.float(), self.normalized_shape,
                            weight, bias, self.eps).type_as(x)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x),
                        None if self.bias is None else self.bias.type_as(x))

    def reset_parameters(self):
        pass


class Embedding(nn.Embedding):
    def reset_parameters(self):
        pass


@torch.jit.script
def gelu(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x ** 2)))


# use jit script to create the IF node
@torch.jit.script_if_tracing
def merge_kv_cache(k: Tensor, v: Tensor, kv_cache: tuple[Tensor, Tensor], use_past: torch.BoolTensor):
    if use_past:
        k_cache, v_cache = kv_cache
        k = torch.cat([k_cache, k], dim=1)
        v = torch.cat([v_cache, v], dim=1)
    kv_cache = (k.detach(), v.detach())
    return k, v, kv_cache


class GLMAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        layer_idx: int,
        dropout_rate = 0.0,
        bias = True,
        pe_2d = True,
        dtype = None,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_head = n_state // n_head
        assert n_state % (n_head * 4) == 0
        self.layer_idx = layer_idx
        self.pe_2d = pe_2d
        self.qkv_proj = Linear(n_state, n_state * 3, bias=bias, dtype=dtype)
        self.o_proj = Linear(n_state, n_state, bias=bias, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: Tensor,
        freqs_cis: tuple[Tensor, ...],
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, ...]] = None,
        use_past = False,
    ):
        '''
        x:
            Shape: (n_batch, n_seq or n_new (using cache), n_state)

        freqs_cis:
            Tuple of (freqs_cis_r, freqs_cis_i) or a tuple of tuple for 2d pe
            Shape: (n_batch, n_seq or n_new (using cache), d_head // 2 or d_head // 4)

        attention_mask:
            0 for no mask, -inf for masked
            Shape: (n_batch, n_seq, n_seq)

        kv_cache:
            Tuple of (k_cache, v_cache)
            Shape: (n_batch, n_seq - n_new, n_head, d_head)
        '''
        n_batch, n_seq, _ = x.shape
        fused_qkv = self.qkv_proj(x).view(n_batch, n_seq, self.n_head, self.d_head * 3)
        # split chunks on reshaped qkv results
        # use torch.split to make onnx simpler
        q, k, v = torch.split(fused_qkv, self.d_head, dim=-1)

        if self.pe_2d: # 2d positional encoding, why not token_type_id embedding?
            emb_dim = self.d_head // 4
            q1_r, q1_i, q2_r, q2_i = torch.split(q, emb_dim, dim=-1)
            k1_r, k1_i, k2_r, k2_i = torch.split(k, emb_dim, dim=-1)
            freqs_cis_0, freqs_cis_1 = freqs_cis
            q1 = apply_rotary_emb(q1_r, q1_i, *freqs_cis_0)
            q2 = apply_rotary_emb(q2_r, q2_i, *freqs_cis_1)
            k1 = apply_rotary_emb(k1_r, k1_i, *freqs_cis_0)
            k2 = apply_rotary_emb(k2_r, k2_i, *freqs_cis_1)
            q = torch.cat([q1, q2], dim=-1)
            k = torch.cat([k1, k2], dim=-1)
        else:
            emb_dim = self.d_head // 2
            q_r, q_i = torch.split(q, emb_dim, dim=-1)
            k_r, k_i = torch.split(k, emb_dim, dim=-1)
            q = apply_rotary_emb(q_r, q_i, *freqs_cis)
            k = apply_rotary_emb(k_r, k_i, *freqs_cis)

        k, v, kv_cache = merge_kv_cache(k, v, kv_cache, use_past)

        scaling_coeff = float(self.layer_idx + 1)
        d_head = q.shape[-1]
        q = q / (math.sqrt(d_head) * scaling_coeff)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        # (n_batch, n_heads, n_new, n_seq)
        qk = torch.matmul(q, k) # / math.sqrt(d_head) # no need to scale again
        if attention_mask is not None:
            qk = qk + attention_mask[:, None, :, :]

        scores = F.softmax(qk.float() * scaling_coeff, dim=-1).type_as(x)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v).permute(0, 2, 1, 3) \
            .contiguous().view(n_batch, n_seq, -1)
        output = self.o_proj(output)

        return output, kv_cache


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout_rate = 0.0,
        bias = True,
        dtype = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w_in = Linear(dim, hidden_dim, bias=bias, dtype=dtype)
        self.w_out = Linear(hidden_dim, dim, bias=bias, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor):
        return self.w_out(self.dropout(gelu(self.w_in(x))))


class GLMBlock(nn.Module):
    def __init__(self, layer_idx: int, config: ChatGLMConfig, dtype=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_layers = config.num_layers
        self.attn_ln = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=dtype)
        self.attn = GLMAttention(
            config.hidden_size, config.num_attention_heads,
            layer_idx, pe_2d=config.position_encoding_2d, dtype=dtype)
        self.ffn_ln = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=dtype)
        self.ffn = FeedForward(
            config.hidden_size, config.inner_hidden_size,
            config.dropout_rate, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        freqs_cis: tuple[Tensor, ...],
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, ...]] = None,
        use_past = False,
    ):
        x = self.attn_ln(x)
        h, kv_cache = self.attn(
            x=x,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_past=use_past,
        )

        alpha = (2 * self.n_layers) ** 0.5
        h = x * alpha + h

        x = self.ffn_ln(h)
        h = self.ffn(x)
        output = x * alpha + h

        return output, kv_cache


class ChatGLMModel(nn.Module):
    def __init__(self, config: ChatGLMConfig, dtype=None):
        super().__init__()
        self.config = config
        self.word_embedding = Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, dtype=dtype
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = torch.nn.ModuleList([
            GLMBlock(layer_idx, config, dtype=dtype) for layer_idx in range(config.num_layers)
        ])
        self.final_ln = LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

        d_freqs_cis = config.hidden_size // config.num_attention_heads
        if config.position_encoding_2d:
            d_freqs_cis //= 2
        freqs_cis_cache_r, freqs_cis_cache_i = \
            precompute_freqs_cis(d_freqs_cis, config.max_sequence_length)

        self.register_buffer("freqs_cis_cache_r",
                             freqs_cis_cache_r.to(dtype=dtype), persistent=False)
        self.register_buffer("freqs_cis_cache_i",
                             freqs_cis_cache_i.to(dtype=dtype), persistent=False)

    @property
    def freqs_cis_cache(self) -> tuple[Tensor, Tensor]:
        return (self.freqs_cis_cache_r, self.freqs_cis_cache_i)

    def prepare_pe_and_mask(
        self,
        input_embeddings: Tensor,
        input_prefix_mask: Tensor,
        has_past_key_values: bool
    ):
        n_batch, n_seq = input_prefix_mask.shape
        n_new = input_embeddings.shape[1]
        device = input_embeddings.device

        # rotary positional encoding
        if self.config.position_encoding_2d:
            prefix_pos = torch.cumsum(input_prefix_mask, dim=1) - 1
            assert (prefix_pos[:, 0] == 0).all(), "At least one prefix token is required"
            suffix_pos = torch.cumsum(-input_prefix_mask + 1, dim=1)
            if has_past_key_values:
                prefix_pos = prefix_pos[:, -n_new:]
                suffix_pos = suffix_pos[:, -n_new:]
            freqs_cis = (
                tuple(F.embedding(prefix_pos, emb) for emb in self.freqs_cis_cache),
                tuple(F.embedding(suffix_pos, emb) for emb in self.freqs_cis_cache),
            )
        else:
            positions = torch.arange(n_seq, device=device).unsqueeze(0).repeat(n_batch, 1)
            if has_past_key_values:
                positions = positions[:, -n_new:]
            freqs_cis = tuple(F.embedding(positions, emb) for emb in self.freqs_cis_cache)

        # causal mask with full prefix attention
        attention_mask = (
            torch.tril(torch.ones((n_batch, n_seq, n_seq), device=device).bool())
            | input_prefix_mask.unsqueeze(1).bool()
        )
        # cast 0 => -inf, 1 => 0
        attention_mask = attention_mask.int().log().type_as(input_embeddings)
        if has_past_key_values:
            attention_mask = attention_mask[:, -n_new:, :].contiguous()

        return freqs_cis, attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        input_prefix_mask: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[Tensor, ...], ...]] = None,
        use_past = True,
        predict_one_token = False,
    ):
        '''
        x:
            Shape: (n_batch, n_seq or n_new (using cache))

        labels:
            Same as x (no shift required) with -100 for prefix and pad tokens

        input_prefix_mask:
            1 for prefix (including [gmask], excluding <bos>), 0 for generated tokens
            Shape: (n_batch, n_seq)
        '''
        if input_embeddings is None:
            assert input_ids is not None, "No input"
            input_embeddings = self.word_embedding(input_ids)
        else:
            assert input_ids is None, "Specify either 'input_ids' or 'input_embeddings'"

        has_past_key_values = past_key_values is not None
        assert not has_past_key_values or len(past_key_values) > 0

        freqs_cis, attention_mask = \
            self.prepare_pe_and_mask(input_embeddings, input_prefix_mask, has_past_key_values)

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
                use_past=use_past and has_past_key_values
            )
            if use_past:
                current_key_values += (kv_cache, )

        h = self.final_ln(h)

        if predict_one_token:
            output = self.lm_head(h[:, -1:, :])
            return None, output, current_key_values

        output = self.lm_head(h)

        if labels is not None:
            n_classes = self.config.vocab_size
            shift_logits = output[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, n_classes), shift_labels.view(-1))
        else:
            loss = None

        return loss, output, current_key_values
