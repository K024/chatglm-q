import torch
from torch import nn, Tensor
from torch.autograd.function import FunctionCtx


try:
    from .triton_ops import (
        check_input,
        dynamic_quant_matmul as _dynamic_quant_matmul_impl,
        dynamic_quant_matmul_transposed as _dynamic_quant_matmul_transposed_impl,
    )
    KERNEL_IMPL = "triton"
except ImportError as e:
    print("Import triton ops failed. Using slower torch fallback.")
    check_input = None
    KERNEL_IMPL = "none"


class DynamicQuantizeMatMul(torch.autograd.Function):
    '''
    ONNXRuntime custom op
    com.microsoft::DynamicQuantizeMatMul

    A: tensor(float) m × k
    B: tensor(int8) k × n
    b_scale: tensor(float) n

    In PyTorch, the weigth is dequantized first.
    '''

    THROW_IF_NOT_USING_TRITON_OPS = False

    @staticmethod
    def forward(ctx: FunctionCtx, A: Tensor, B: Tensor, b_scale: Tensor):
        # 'A' must be saved to get grad
        ctx.save_for_backward(A, B, b_scale)
        if check_input and check_input(A):
            out = _dynamic_quant_matmul_impl(A, B, b_scale)
        else:
            out = A.matmul(B * b_scale)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor):
        A, B, b_scale = ctx.saved_tensors

        grad_A = None
        if ctx.needs_input_grad[0]:
            if check_input and check_input(A):
                grad_A = _dynamic_quant_matmul_transposed_impl(grad_out, B, b_scale)
            else:
                grad_A = grad_out.matmul(B.t() * b_scale[:, None])

        return grad_A, None, None
    
    ONNX_CPU_ONLY = True

    @staticmethod
    def symbolic(g: torch.Graph, A, B, b_scale) -> torch.Value:
        if DynamicQuantizeMatMul.ONNX_CPU_ONLY:
            # return g.op("com.microsoft::DynamicQuantizeMatMul", A, B, b_scale)
            A_quant, A_scale, A_zero = g.op("DynamicQuantizeLinear", A, outputs=3)
            C = g.op("MatMulInteger", A_quant, B, A_zero, torch.tensor(0, dtype=torch.int8))
            C = g.op("Cast", C, to_i=1) # TensorProto.DataType.FLOAT=1
            return g.op("Mul", C, g.op("Mul", A_scale, b_scale))
        else:
            # is unstable on CUDA and produces NaN
            A_scale = g.op("Div", g.op("ReduceMax", g.op("Abs", A), keepdims_i=0), torch.tensor(127, dtype=torch.float32))
            A_quant = g.op("QuantizeLinear", A, A_scale, torch.tensor(0, dtype=torch.int8))
            C = g.op("MatMulInteger", A_quant, B, torch.tensor(0, dtype=torch.int8), torch.tensor(0, dtype=torch.int8))
            C = g.op("Cast", C, to_i=1) # TensorProto.DataType.FLOAT=1
            return g.op("Mul", C, g.op("Mul", A_scale, b_scale))


def dynamic_quant_matmul(A: Tensor, B: torch.CharTensor, b_scale: Tensor) -> Tensor:
    return DynamicQuantizeMatMul.apply(A, B, b_scale)


class DynamicQuantizeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), device=device, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, device=device, dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_buffer('bias', None)

    def forward(self, input: Tensor):
        out = dynamic_quant_matmul(input, self.weight.t(), self.weight_scale)
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
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def reset_parameters(self):
        pass


class QEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight", torch.empty((num_embeddings, embedding_dim), device=device, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(embedding_dim, device=device, dtype=dtype))

    def forward(self, input: Tensor):
        embeddings = nn.functional.embedding(input, self.weight)
        return embeddings * self.weight_scale

    @torch.no_grad()
    def apply_weights_(self, q_weight: Tensor, scale: Tensor):
        self.weight.copy_(q_weight)
        self.weight_scale.copy_(scale)

    def extra_repr(self) -> str:
        return 'num_embeddings={}, embedding_dim={}'.format(
            self.num_embeddings, self.embedding_dim)

    def reset_parameters(self):
        pass
