import torch
from torch import Tensor

import triton
import triton.language as tl
# from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def check_input(a: torch.Tensor):
    return a.get_device() >= 0


@triton.autotune(
    configs=[
        # multiple configs not working for triton==2.0.0.post1
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _dynamic_quant_matmul_kernel(
    A, B, B_scale, C, M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn, stride_bscale,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    '''
    A:        (M, K)  *float
    B:        (K, N)  *int8
    B_scale:  (N)     *float
    C:        (M, N)  *float
    '''
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
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
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    B_scale = B_scale + (rbn[None, :] * stride_bscale)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = tl.load(B_scale)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.)
        b = b * scale
        acc += tl.dot(a, b, allow_tf32=allow_tf32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def dynamic_quant_matmul(a: Tensor, b: Tensor, b_scale: Tensor, allow_tf32: bool=None):
    '''
    a:        (M, K)  *float
    b:        (K, N)  *int8
    b_scale:  (N)     *float
    returns:  (M, N)  *float
    '''
    # checks constraints
    output_shape = (*a.shape[:-1], b.shape[1])
    a = a.flatten(0, -2)
    assert len(b.shape) == 2
    assert len(b_scale.shape) == 1
    assert a.shape[1] == b.shape[0]
    assert b.shape[1] == b_scale.shape[0]
    assert b.dtype == torch.int8
    assert a.dtype == b_scale.dtype
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
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # launch kernel
    if allow_tf32 is None:
        allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    with torch.cuda.device(a.device):
        _dynamic_quant_matmul_kernel[grid](
            a, b, b_scale, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), b_scale.stride(0),
            c.stride(0), c.stride(1),
            allow_tf32=allow_tf32,
        )
        return c.reshape(output_shape)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _dynamic_quant_matmul_transposed_kernel(
    A, B_T, B_scale, C, M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk, stride_bscale,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    '''
    A:        (M, K)  *float
    B_T:      (N, K)  *int8  (transposed)
    B_scale:  (K)     *float (transposed scale)
    C:        (M, N)  *float
    '''
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
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
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B_T = B_T + (rbn[:, None] * stride_bn + rk[None, :] * stride_bk)
    B_scale = B_scale + (rk[None, :] * stride_bscale)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B_T)
            scale = tl.load(B_scale)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B_T, mask=rk[None, :] < k_remaining, other=0.)
            scale = tl.load(B_scale, mask=rk[None, :] < k_remaining, other=1.)
        b = tl.trans(b * scale)
        acc += tl.dot(a, b, allow_tf32=allow_tf32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B_T += BLOCK_K * SPLIT_K * stride_bk
        B_scale += BLOCK_K * SPLIT_K * stride_bscale
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def dynamic_quant_matmul_transposed(a: Tensor, b_T: Tensor, b_scale: Tensor, allow_tf32: bool=None):
    '''
    a:        (M, K)  float
    b_T:      (N, K)  int8  (transposed)
    b_scale:  (K)     float (transposed scale)
    returns:  (M, N)  float
    '''
    # checks constraints
    output_shape = (*a.shape[:-1], b_T.shape[0])
    a = a.flatten(0, -2)
    assert len(b_T.shape) == 2
    assert len(b_scale.shape) == 1
    assert a.shape[1] == b_T.shape[1]
    assert b_T.shape[1] == b_scale.shape[0]
    assert b_T.dtype == torch.int8
    assert a.dtype == b_scale.dtype
    assert a.get_device() >= 0
    assert b_T.get_device() == a.get_device(), f"{b_T.device=}, {a.device=}"
    assert b_scale.get_device() == a.get_device(), f"{b_scale.device=}, {a.device=}"
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b_T.stride(0) > 1 and b_T.stride(1) > 1:
        b_T = b_T.contiguous()
    # allocates output
    M, K = a.shape
    N, _ = b_T.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # launch kernel
    if allow_tf32 is None:
        allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    with torch.cuda.device(a.device):
        _dynamic_quant_matmul_transposed_kernel[grid](
            a, b_T, b_scale, c, M, N, K,
            a.stride(0), a.stride(1),
            b_T.stride(0), b_T.stride(1), b_scale.stride(0),
            c.stride(0), c.stride(1),
            allow_tf32=allow_tf32,
        )
        return c.reshape(output_shape)
