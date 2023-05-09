
import torch
from torch import Tensor

import triton
import triton.language as tl
# from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def check_input(a: torch.Tensor):
    return a.get_device() >= 0


def is_power_of_two(n: int):
    return (n & (n-1) == 0) and n != 0


@triton.autotune(
    configs=[
        # multiple configs not working for triton==2.0.0.post1
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dynamic_quant_matmul_s4_kernel(
    A, B, B_scale, C, M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bscale_g, stride_bscale_n,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, GROUP_K: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    '''
    A:        (M, K)     *float
    B:        (K//2, N)  *uint8
    B_scale:  (G, N)     *float
    C:        (M, N)     *float

    requirements: K // G == GROUP_K, GROUP_K % BLOCK_K == 0
    '''
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + ((rk[:, None] // 2) * stride_bk + rbn[None, :] * stride_bn)
    B_scale = B_scale + (0 * stride_bscale_g + rbn[None, :] * stride_bscale_n)
    quant_group_by_k = (GROUP_K // BLOCK_K)
    B_shift = ((rk % 2) * 4)[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)

        scale = tl.load(B_scale)
        b = ((b >> B_shift) & 0xF).to(tl.int8)
        b = (b - 0x8) * scale

        acc += tl.dot(a, b, allow_tf32=allow_tf32)
        A += BLOCK_K * stride_ak
        B += (BLOCK_K // 2) * stride_bk
        if (k + 1) % quant_group_by_k == 0:
            B_scale += stride_bscale_g
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


def dynamic_quant_matmul_s4(a: Tensor, b: Tensor, b_scale: Tensor, allow_tf32: bool=None):
    '''
    A:        (M, K)     *float
    B:        (K//2, N)  *uint8
    B_scale:  (G, N)     *float
    C:        (M, N)     *float

    requirements: K // G == GROUP_K, GROUP_K % BLOCK_K == 0
    '''
    # checks constraints
    output_shape = (*a.shape[:-1], b.shape[1])
    a = a.flatten(0, -2)
    assert len(b.shape) == 2
    assert len(b_scale.shape) == 2
    assert a.shape[1] == b.shape[0] * 2
    assert b.shape[1] == b_scale.shape[1]
    assert b.dtype == torch.uint8
    assert a.dtype == b_scale.dtype
    assert b.shape[0] % b_scale.shape[0] == 0
    assert a.get_device() >= 0
    assert b.get_device() == a.get_device(), f"{b.device=}, {a.device=}"
    assert b_scale.get_device() == a.get_device(), f"{b_scale.device=}, {a.device=}"
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # allocates output
    M, K = a.shape
    G, N = b_scale.shape
    GROUP_K = K // G
    BLOCK_K = min(64, GROUP_K)
    assert is_power_of_two(BLOCK_K)
    assert is_power_of_two(GROUP_K)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # launch kernel
    if allow_tf32 is None:
        allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    with torch.cuda.device(a.device):
        _dynamic_quant_matmul_s4_kernel[grid](
            a, b, b_scale, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            b_scale.stride(0), b_scale.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_K=BLOCK_K, GROUP_K=GROUP_K,
            allow_tf32=allow_tf32,
        )
        return c.reshape(output_shape)


@triton.autotune(
    configs=[
        # multiple configs not working for triton==2.0.0.post1
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dynamic_quant_matmul_transposed_s4_kernel(
    A, B, B_scale, C, M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_bscale_g, stride_bscale_k,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, GROUP_N: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    '''
    A:        (M, K)     *float
    B:        (N//2, K)  *uint8
    B_scale:  (G, K)     *float
    C:        (M, N)     *float

    requirements: N // G == GROUP_N, GROUP_N % BLOCK_N == 0, K % BLOCK_K == 0
    '''
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + ((rbn[:, None] // 2) * stride_bn + rk[None, :] * stride_bk)
    quant_group = pid_n // (GROUP_N // BLOCK_N)
    B_scale = B_scale + (quant_group * stride_bscale_g + rk[None, :] * stride_bscale_k)
    B_shift = ((rbn % 2) * 4)[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)

        scale = tl.load(B_scale)
        b = ((b >> B_shift) & 0xF).to(tl.int8)
        b = tl.trans((b - 0x8) * scale)

        acc += tl.dot(a, b, allow_tf32=allow_tf32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
        B_scale += BLOCK_K * stride_bscale_k
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


def dynamic_quant_matmul_transposed_s4(a: Tensor, b: Tensor, b_scale: Tensor, allow_tf32: bool=None):
    '''
    A:        (M, K)     *float
    B:        (N//2, K)  *uint8
    B_scale:  (G, K)     *float
    C:        (M, N)     *float

    requirements: N // G == GROUP_N, GROUP_N % BLOCK_N == 0, K % BLOCK_K == 0
    '''
    # checks constraints
    output_shape = (*a.shape[:-1], b.shape[0] * 2)
    a = a.flatten(0, -2)
    assert len(b.shape) == 2
    assert len(b_scale.shape) == 2
    assert a.shape[1] == b.shape[1]
    assert b.shape[1] == b_scale.shape[1]
    assert b.dtype == torch.uint8
    assert a.dtype == b_scale.dtype
    assert b.shape[0] % b_scale.shape[0] == 0
    assert a.get_device() >= 0
    assert b.get_device() == a.get_device(), f"{b.device=}, {a.device=}"
    assert b_scale.get_device() == a.get_device(), f"{b_scale.device=}, {a.device=}"
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # allocates output
    M, K = a.shape
    G, _ = b_scale.shape
    N = b.shape[0] * 2
    GROUP_N = N // G
    BLOCK_N = min(64, GROUP_N)
    assert is_power_of_two(K)
    assert is_power_of_two(BLOCK_N)
    assert is_power_of_two(GROUP_N)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # launch kernel
    if allow_tf32 is None:
        allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    with torch.cuda.device(a.device):
        _dynamic_quant_matmul_transposed_s4_kernel[grid](
            a, b, b_scale, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            b_scale.stride(0), b_scale.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_N=BLOCK_N, GROUP_N=GROUP_N,
            allow_tf32=allow_tf32,
        )
        return c.reshape(output_shape)
