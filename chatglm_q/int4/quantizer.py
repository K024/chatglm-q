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
def clamp_to_quantize_grid(x: Tensor, scale: Tensor) -> Tensor:
    q = torch.clamp(torch.round(x / scale), -max_q_int4, max_q_int4)
    return scale * q


@torch.no_grad()
def quantize_with_scale(x: Tensor, scale: Tensor) -> Tensor:
    assert len(x.shape) == 2
    assert len(scale.shape) == 2
    K, N = x.shape
    G, _ = scale.shape
    assert K % G == 0
    GROUP_K = K // G
    x = x.reshape((G, GROUP_K, N))
    scale = scale.reshape((G, 1, N))
    x = torch.clamp(torch.round(x / scale), -max_q_int4, max_q_int4)
    x = (x + 0x8).to(torch.uint8)
    # pack
    x = x.reshape((K, N))
    x = (x[::2, :] & 0xF) | ((x[1::2, :] & 0xF) << 4)
    return x


@torch.no_grad()
def get_quant_int4_linear(layer: nn.Linear, group_size=qlinear.DEFAULT_GROUP_SIZE):
    assert isinstance(layer, nn.Linear)
    q_weight, scale = quantize_int4(layer.weight.t(), group_size)

    qlinear = DynamicQuantizeLinear(layer.in_features, layer.out_features, layer.bias is not None, group_size)
    qlinear.apply_weights_(q_weight, scale, layer.bias)

    return qlinear


@torch.no_grad()
def get_quant_embedding(layer: nn.Embedding, group_size=qlinear.DEFAULT_GROUP_SIZE):
    assert isinstance(layer, nn.Embedding)
    q_weight, scale = quantize_int4(layer.weight, group_size)

    qembedding = QEmbedding(layer.num_embeddings, layer.embedding_dim)
    qembedding.apply_weights_(q_weight, scale)

    return qembedding


class GPTQLinearQuantizer():
    '''
    GPTQ quantization, see:
        paper: https://arxiv.org/abs/2210.17323
        code: https://github.com/IST-DASLab/gptq
        license: Apache License 2.0 https://github.com/IST-DASLab/gptq/blob/main/LICENSE
    '''

    def __init__(self, layer: nn.Module):
        assert isinstance(layer, nn.Linear)
        self.layer = layer
        self.hook = layer.register_forward_hook(self.forward_hook)
        # output_dim, input_dim
        self.n_rows, self.n_columns = layer.weight.shape
        self.hessian = torch.zeros((self.n_columns, self.n_columns), device=layer.weight.device)
        self.n_samples = 0
        self.debug_input = None
        self.debug_output = None

    @torch.no_grad()
    def forward_hook(self, module: nn.Module, inputs: tuple[Tensor], output: Tensor):
        input, = inputs
        if len(input.shape) > 2:
            input = input.flatten(0, -2)

        self.debug_input = input.detach()
        self.debug_output = output.detach()

        new_samples, d_hidden = input.shape
        assert d_hidden == self.n_columns

        input = input.t()
        self.hessian *= self.n_samples / (self.n_samples + new_samples)
        self.n_samples += new_samples
        input = math.sqrt(2 / self.n_samples) * input.float()

        self.hessian += input.matmul(input.t())

    def remove_hook(self):
        self.hook.remove()

    @torch.no_grad()
    def quantize_weight(self, blocksize=128, groupsize=qlinear.DEFAULT_GROUP_SIZE, percdamp=.01):
        assert self.n_samples > 0
        assert blocksize % groupsize == 0, f"{blocksize=}, {groupsize=}"
        assert self.n_columns % groupsize == 0, f"{self.n_columns=}, {groupsize=}"

        hessian = self.hessian
        weight = self.layer.weight.clone()
        scales = []

        dead_rows = torch.diag(hessian) == 0
        hessian[dead_rows, dead_rows] = 1
        weight[:, dead_rows] = 0

        quant_losses = torch.zeros_like(weight)
        grid_weight = torch.zeros_like(weight)

        damp = percdamp * torch.mean(torch.diag(hessian))
        diag = torch.arange(self.n_columns, device=weight.device)
        hessian[diag, diag] += damp
        hessian_inv = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
        hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)

        assert not hessian_inv.isnan().any()

        for i1 in range(0, self.n_columns, blocksize):
            i2 = min(i1 + blocksize, self.n_columns)

            weight_block = weight[:, i1:i2].clone()
            quant_block = torch.zeros_like(weight_block)
            err_block = torch.zeros_like(weight_block)
            losses_block = torch.zeros_like(weight_block)
            h_inv_block = hessian_inv[i1:i2, i1:i2]

            for j in range(i2 - i1):
                w = weight_block[:, j]
                d = h_inv_block[j, j]

                if j % groupsize == 0:
                    _, scale = quantize_int4(weight_block[:, j:j + groupsize].t(), groupsize)
                    assert scale.shape == (1, self.n_rows)
                    scales.append(scale)

                q = clamp_to_quantize_grid(w, scale[0])

                quant_block[:, j] = q
                losses_block[:, j] = (w - q) ** 2 / d ** 2

                err = (w - q) / d
                weight_block[:, j:] -= err.unsqueeze(1).matmul(h_inv_block[j, j:].unsqueeze(0))
                err_block[:, j] = err

            grid_weight[:, i1:i2] = quant_block
            quant_losses[:, i1:i2] = losses_block / 2

            weight[:, i2:] -= err_block.matmul(hessian_inv[i1:i2, i2:])

        quant_out = nn.functional.linear(self.debug_input, grid_weight, self.layer.bias)
        debug_loss = torch.sum((quant_out - self.debug_output) ** 2)

        scale = torch.cat(scales, dim=0)

        return grid_weight, scale, quant_losses, debug_loss

    @torch.no_grad()
    def get_quantized_linear(self, blocksize=128, groupsize=qlinear.DEFAULT_GROUP_SIZE, percdamp=.01, pring_loss=False):
        grid_weight, scale, quant_losses, debug_loss = self.quantize_weight(blocksize, groupsize, percdamp)

        if pring_loss:
            print(f"{quant_losses.sum()=} {debug_loss=}")

        original = self.layer
        modified = DynamicQuantizeLinear(original.in_features, original.out_features, original.bias is not None, groupsize)

        q_weight = quantize_with_scale(grid_weight.t(), scale)
        modified.apply_weights_(q_weight, scale, original.bias)

        return modified
