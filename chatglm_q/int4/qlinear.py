import torch
from torch import nn, Tensor
from torch.autograd.function import FunctionCtx

DEFAULT_GROUP_SIZE = 32

try:
    from .triton_ops import (
        dynamic_quant_matmul_s4 as _dynamic_quant_matmul_impl,
        dynamic_quant_matmul_transposed_s4 as _dynamic_quant_matmul_transposed_impl,
    )
except ImportError as e:
    print("Import triton ops failed. Using slower torch fallback.")
    _dynamic_quant_matmul_impl = None
    _dynamic_quant_matmul_transposed_impl = None


@torch.no_grad()
def unpack_int4(x: torch.Tensor, x_scale: torch.Tensor):
    # shape
    K = x.shape[0] * 2
    G, N = x_scale.shape
    assert x.shape[1] == N
    assert K % G == 0
    GROUP_K = K // G
    # unpack
    shifts = torch.tensor([[0], [4]]).type_as(x).repeat((K // 2, 1))
    x = x.reshape((K // 2, 1, N)).repeat((1, 2, 1)).reshape((K, N))
    x = ((x >> shifts) & 0xF).to(torch.int8) - 0x8
    x = x.reshape((G, GROUP_K, N)) * x_scale[:, None, :]
    return x.reshape((K, N))


class DynamicQuantizeMatMul(torch.autograd.Function):
    '''
    A: tensor(float) m × k
    B: tensor(int8) k//2 × n
    b_scale: tensor(float) g × n
    '''

    THROW_IF_NOT_USING_TRITON_OPS = False

    @staticmethod
    def forward(ctx: FunctionCtx, A: Tensor, B: Tensor, b_scale: Tensor):
        # 'A' must be saved to get grad
        ctx.save_for_backward(A, B, b_scale)
        if A.device.type == "cuda" and _dynamic_quant_matmul_impl is not None:
            out = _dynamic_quant_matmul_impl(A, B, b_scale)
        elif DynamicQuantizeMatMul.THROW_IF_NOT_USING_TRITON_OPS:
            raise NotImplementedError()
        else:
            out = A.matmul(unpack_int4(B, b_scale))
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor):
        A, B, b_scale = ctx.saved_tensors

        grad_A = None
        if ctx.needs_input_grad[0]:
            if A.device.type == "cuda" and _dynamic_quant_matmul_transposed_impl is not None:
                grad_A = _dynamic_quant_matmul_transposed_impl(grad_out, B, b_scale)
            elif DynamicQuantizeMatMul.THROW_IF_NOT_USING_TRITON_OPS:
                raise NotImplementedError()
            else:
                grad_A = grad_out.matmul(unpack_int4(B, b_scale).t())

        return grad_A, None, None

    @staticmethod
    def symbolic(g: torch.Graph, A, B, b_scale) -> torch.Value:
        raise NotImplementedError()


def dynamic_quant_matmul(A: Tensor, B: torch.CharTensor, b_scale: Tensor) -> Tensor:
    return DynamicQuantizeMatMul.apply(A, B, b_scale)


class DynamicQuantizeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True, group_size=DEFAULT_GROUP_SIZE, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert in_features % group_size == 0
        assert group_size % 32 == 0
        self.group_size = group_size
        self.groups = in_features // group_size
        self.register_buffer("weight", torch.empty((in_features//2, out_features), device=device, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.empty((self.groups, out_features), device=device, dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_buffer('bias', None)

    def forward(self, input: Tensor):
        out = dynamic_quant_matmul(input, self.weight, self.weight_scale)
        if self.bias is not None:
            out += self.bias
        return out

    @torch.no_grad()
    def apply_weights_(self, q_weight: Tensor, scale: Tensor, bias: Tensor = None):
        self.weight.copy_(q_weight)
        self.weight_scale.copy_(scale)
        if bias is not None:
            self.bias.copy_(bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, group_size={}, bias={}'.format(
            self.in_features, self.out_features, self.group_size, self.bias is not None)

    def reset_parameters(self):
        pass


class QEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, group_size=DEFAULT_GROUP_SIZE, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        assert num_embeddings % group_size == 0
        assert group_size % 32 == 0
        self.group_size = group_size
        self.groups = num_embeddings // group_size
        self.register_buffer("weight", torch.empty((num_embeddings//2, embedding_dim), device=device, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.empty((self.groups, embedding_dim), device=device, dtype=dtype))

    def forward(self, input: Tensor):
        group_idx = input // self.group_size
        embed_idx = input // 2
        shifts = ((input % 2) * 4)[..., None]
        scales = nn.functional.embedding(group_idx, self.weight_scale)
        embeddings = nn.functional.embedding(embed_idx, self.weight)
        embeddings = ((embeddings >> shifts) & 0xF).to(torch.int8)
        embeddings = (embeddings - 0x8) * scales
        return embeddings

    @torch.no_grad()
    def apply_weights_(self, q_weight: Tensor, scale: Tensor):
        self.weight.copy_(q_weight)
        self.weight_scale.copy_(scale)

    def extra_repr(self) -> str:
        return 'num_embeddings={}, embedding_dim={}, group_size={}'.format(
            self.num_embeddings, self.embedding_dim, self.group_size)

    def reset_parameters(self):
        pass
