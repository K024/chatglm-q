import math
import torch
from torch import nn, Tensor
from . import qlinear
from .qlinear import DynamicQuantizeLinear, QEmbedding


max_q_int4 = 2 ** (4 - 1) - 1
assert 7 == max_q_int4


@torch.no_grad()
def quantize_int4(x: torch.Tensor, GROUP_K = qlinear.DEFAULT_GROUP_SIZE):
    '''
    inputs: for weight (in_dim, out_dim)
    '''
    assert len(x.shape) == 2
    K, N = x.shape
    assert K % GROUP_K == 0
    G = K // GROUP_K
    x = x.reshape((G, GROUP_K, N))
    w_max, _ = torch.max(x.abs(), dim=1, keepdim=True)
    scale = torch.clamp(w_max / max_q_int4, min=1e-10)
    x = torch.clamp(torch.round(x / scale), -max_q_int4, max_q_int4)
    x = (x + 0x8).to(torch.uint8)
    # pack
    x = x.reshape((K, N))
    x = (x[::2, :] & 0xF) | ((x[1::2, :] & 0xF) << 4)
    return x.contiguous(), scale.reshape((G, N))


@torch.no_grad()
def clamp_to_quantize_grid(x: Tensor, scale: Tensor, max_q: int) -> Tensor:
    assert len(x.shape) == 2
    assert len(scale.shape) == 2
    K, N = x.shape
    G, _ = scale.shape
    assert K % G == 0
    GROUP_K = K // G
    x = x.reshape((G, GROUP_K, N))
    scale = scale.reshape((G, 1, N))
    q = torch.clamp(torch.round(x / scale), -max_q, max_q)
    return scale * q


@torch.no_grad()
def quantize_with_scale(x: Tensor, scale: Tensor, max_q: int) -> Tensor:
    assert len(x.shape) == 2
    assert len(scale.shape) == 2
    K, N = x.shape
    G, _ = scale.shape
    assert K % G == 0
    GROUP_K = K // G
    x = x.reshape((G, GROUP_K, N))
    scale = scale.reshape((G, 1, N))
    x = torch.clamp(torch.round(x / scale), -max_q, max_q)
    # pack
    x = x.reshape((K, N))
    x = (x[::2, :] & 0xF) | ((x[1::2, :] & 0xF) << 4)
    return x


@torch.no_grad()
def get_quant_int4_linear(layer: nn.Linear):
    assert isinstance(layer, nn.Linear)
    q_weight, scale = quantize_int4(layer.weight.t())

    qlinear = DynamicQuantizeLinear(layer.in_features, layer.out_features, layer.bias is not None)
    qlinear.apply_weights_(q_weight, scale, layer.bias)

    return qlinear


@torch.no_grad()
def get_quant_embedding(layer: nn.Embedding):
    assert isinstance(layer, nn.Embedding)
    q_weight, scale = quantize_int4(layer.weight)

    qembedding = QEmbedding(layer.num_embeddings, layer.embedding_dim)
    qembedding.apply_weights_(q_weight, scale)

    return qembedding
