"""
FlashAttention-3 forward pass in Triton Gluon (Hopper SM90).

Implements the core FlashAttention-3 techniques from:
  "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
  (Shah et al., 2024)

Built with explicit Hopper primitives from `triton.experimental.gluon`
(following `triton/python/tutorials/gluon/04-tma.py`, `05-wgmma.py`, and
`08-warp-specialization.py`):
  1. TMA + Swizzled SMEM (`04-tma.py`):
     Host-side `TensorDescriptor.from_tensor` with `NVMMASharedLayout`,
     `tma.async_load` / `tma.async_store`, and explicit `mbarrier` tracking.
  2. Asynchronous WGMMA (`05-wgmma.py`):
     `warpgroup_mma(..., is_async=True)` with `NVMMADistributedLayout`,
     zero-copy SMEM transpose (`k_buf.permute((1, 0))`), and `DotOperandLayout`
     so `P_ij` stays in registers for `O_i += P_ij V_j`.
  3. 2-Stage GEMM-Softmax Pipelining (FA3 Algorithm 2):
     Issues `GEMM0(j)` (`Q_i K_j^T`) and `GEMM1(j-1)` (`P_{j-1} V_{j-1}`)
     back-to-back, waits on `warpgroup_mma_wait(num_outstanding=1)` for `S_j`,
     and overlaps online softmax (`MUFU.EX2`) with `GEMM1(j-1)` on Tensor Cores.
  4. Producer-Consumer Warp Specialization (`08-warp-specialization.py`):
     `gl.warp_specialize` pairs a 1-warp (24-reg) TMA producer streaming `K, V`
     into circular SMEM buffers with a 4-warp WGMMA consumer warpgroup.

Note: Requires an NVIDIA Hopper (SM90) GPU and Triton with `triton.experimental.gluon`.
"""

import math
import time
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

LOG2_E = 1.4426950408889634
LN_2 = gl.constexpr(0.6931471805599453)


def is_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


@gluon.constexpr_function
def wgmma_layout(dtype, block_n):
    """Single-warpgroup (4-warp) Hopper WGMMA output layout for [B_r, block_n]."""
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, min(block_n, 256), 256 // dtype.primitive_bitwidth],
    )


@gluon.jit
def _apply_mask(s_cur, offset_m, offset_n, N, IS_CAUSAL: gl.constexpr):
    if IS_CAUSAL:
        mask = (offset_m[:, None] >= offset_n[None, :]) & (offset_n[None, :] < N)
    else:
        mask = offset_n[None, :] < N
    return gl.where(mask, s_cur, -float("inf"))


# ===-----------------------------------------------------------------------===#
# Producer Partition: 1-Warp TMA Loader
# ===-----------------------------------------------------------------------===#


@gluon.jit
def fa3_load_partition(desc_k, desc_v, bufs, bars, num_kv_blocks, B_c: gl.constexpr):
    k_bufs, v_bufs = bufs
    k_empty_bars, k_ready_bars, v_empty_bars, v_ready_bars = bars
    num_stages: gl.constexpr = k_bufs.type.shape[0]

    for j in range(num_kv_blocks):
        idx = j % num_stages
        phase = (j // num_stages) & 1
        start_n = j * B_c

        # mbarriers start with phase 1 complete -> producer waits on phase ^ 1
        mbarrier.wait(k_empty_bars.index(idx), phase ^ 1)
        mbarrier.expect(k_ready_bars.index(idx), desc_k.block_type.nbytes)
        tma.async_load(desc_k, [start_n, 0], k_ready_bars.index(idx), k_bufs.index(idx))

        mbarrier.wait(v_empty_bars.index(idx), phase ^ 1)
        mbarrier.expect(v_ready_bars.index(idx), desc_v.block_type.nbytes)
        tma.async_load(desc_v, [start_n, 0], v_ready_bars.index(idx), v_bufs.index(idx))


# ===-----------------------------------------------------------------------===#
# Consumer Partition: 4-Warp WGMMA + 2-Stage GEMM-Softmax Pipelining + Epilogue
# ===-----------------------------------------------------------------------===#


@gluon.jit
def fa3_compute_partition(
        q_smem, desc_o, L, bufs, bars, off_m, N, num_kv_blocks, softmax_scale_log2,
        B_r: gl.constexpr, B_c: gl.constexpr, D: gl.constexpr, IS_CAUSAL: gl.constexpr,
):
    k_bufs, v_bufs = bufs
    k_empty_bars, k_ready_bars, v_empty_bars, v_ready_bars = bars
    dtype: gl.constexpr = q_smem.dtype
    num_stages: gl.constexpr = k_bufs.type.shape[0]

    # Explicit WGMMA layouts for S_ij ([B_r, B_c]), O_i ([B_r, D]), and P_ij in registers
    s_layout: gl.constexpr = wgmma_layout(dtype, B_c)
    o_layout: gl.constexpr = wgmma_layout(dtype, D)
    p_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=o_layout, k_width=32 // dtype.primitive_bitwidth
    )
    o_row_layout: gl.constexpr = gl.SliceLayout(1, o_layout)
    l_store_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])

    offset_m = off_m + gl.arange(0, B_r, layout=gl.SliceLayout(1, s_layout))
    offset_n_base = gl.arange(0, B_c, layout=gl.SliceLayout(0, s_layout))
    s_zero = gl.zeros((B_r, B_c), dtype=gl.float32, layout=s_layout)
    o_acc = gl.zeros((B_r, D), dtype=gl.float32, layout=o_layout)

    # --- Prologue (j = 0): S_0 = Q_i K_0^T and initial softmax P_0 ---
    mbarrier.wait(k_ready_bars.index(0), 0)
    s_acc = warpgroup_mma(q_smem, k_bufs.index(0).permute((1, 0)), s_zero, is_async=True, use_acc=False)
    s_cur = warpgroup_mma_wait(num_outstanding=0, deps=(s_acc,))
    mbarrier.arrive(k_empty_bars.index(0), count=1)

    s_cur = s_cur * softmax_scale_log2
    if (IS_CAUSAL and (B_c > off_m)) or (B_c > N):
        s_cur = _apply_mask(s_cur, offset_m, offset_n_base, N, IS_CAUSAL)

    m_i = gl.max(s_cur, axis=1)
    p_cur = gl.exp2(s_cur - m_i[:, None])
    l_i = gl.sum(p_cur, axis=1)
    p_reg = gl.convert_layout(p_cur.to(dtype), p_reg_layout)
    mbarrier.wait(v_ready_bars.index(0), 0)

    # --- Main 2-Stage Pipelined Loop: overlap softmax(S_j) with GEMM1(j-1) ---
    for j in range(1, num_kv_blocks):
        idx = j % num_stages
        phase = (j // num_stages) & 1
        prev_idx = (j - 1) % num_stages
        start_n = j * B_c

        # Issue GEMM0(j) and GEMM1(j-1) back-to-back asynchronously
        mbarrier.wait(k_ready_bars.index(idx), phase)
        s_acc = warpgroup_mma(q_smem, k_bufs.index(idx).permute((1, 0)), s_zero, is_async=True, use_acc=False)
        o_async = warpgroup_mma(p_reg, v_bufs.index(prev_idx), o_acc, is_async=True, use_acc=(j > 1))

        # Wait for GEMM0(j) while GEMM1(j-1) continues on Tensor Cores
        s_cur = warpgroup_mma_wait(num_outstanding=1, deps=(s_acc,))
        mbarrier.arrive(k_empty_bars.index(idx), count=1)

        # Online softmax on S_j (overlapped with GEMM1(j-1))
        s_cur = s_cur * softmax_scale_log2
        if (IS_CAUSAL and (start_n + B_c > off_m)) or (start_n + B_c > N):
            s_cur = _apply_mask(s_cur, offset_m, start_n + offset_n_base, N, IS_CAUSAL)

        m_ij = gl.max(s_cur, axis=1)
        m_i_new = gl.maximum(m_i, m_ij)
        alpha = gl.exp2(m_i - m_i_new)
        p_cur = gl.exp2(s_cur - m_i_new[:, None])
        l_i = l_i * alpha + gl.sum(p_cur, axis=1)
        m_i = m_i_new
        p_reg = gl.convert_layout(p_cur.to(dtype), p_reg_layout)

        # Finish GEMM1(j-1), release V_{j-1}, rescale O_i, and acquire V_j
        o_acc = warpgroup_mma_wait(num_outstanding=0, deps=(o_async,))
        mbarrier.arrive(v_empty_bars.index(prev_idx), count=1)
        o_acc = o_acc * gl.convert_layout(alpha, o_row_layout)[:, None]
        mbarrier.wait(v_ready_bars.index(idx), phase)

    # --- Epilogue: final GEMM1, normalize O_i, TMA store ---
    last_idx = (num_kv_blocks - 1) % num_stages
    o_async = warpgroup_mma(p_reg, v_bufs.index(last_idx), o_acc, is_async=True, use_acc=(num_kv_blocks > 1))
    o_acc = warpgroup_mma_wait(num_outstanding=0, deps=(o_async,))
    mbarrier.arrive(v_empty_bars.index(last_idx), count=1)

    o_acc = o_acc / gl.convert_layout(l_i, o_row_layout)[:, None]
    o_smem = gl.allocate_shared_memory(dtype, desc_o.block_type.shape, desc_o.layout)
    o_smem.store(o_acc.to(dtype))
    fence_async_shared()
    tma.async_store(desc_o, [off_m, 0], o_smem)
    tma.store_wait(pendings=0)

    l_i_log = (m_i + gl.log2(l_i)) * LN_2
    l_offs = off_m + gl.arange(0, B_r, layout=l_store_layout)
    gl.store(L + l_offs, gl.convert_layout(l_i_log, l_store_layout), mask=l_offs < N)


# ===-----------------------------------------------------------------------===#
# Top-Level Warp-Specialized Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def flash_attention_3_gluon_kernel(
        desc_q, desc_k, desc_v, desc_o, L, softmax_scale_log2,
        NUM_STAGES: gl.constexpr, IS_CAUSAL: gl.constexpr,
):
    B_r: gl.constexpr = desc_q.block_type.shape[0]
    B_c: gl.constexpr = desc_k.block_type.shape[0]
    D: gl.constexpr = desc_q.block_type.shape[1]
    dtype: gl.constexpr = desc_q.dtype
    N = desc_q.shape[0]

    # Reverse tile order for causal attention so longest tiles launch first
    q_tile = (gl.num_programs(0) - 1 - gl.program_id(0)) if IS_CAUSAL else gl.program_id(0)
    off_m = q_tile * B_r
    num_kv_blocks = gl.cdiv(gl.minimum((q_tile + 1) * B_r, N) if IS_CAUSAL else N, B_c)

    # Load Q_i once into SMEM before warp specialization
    q_smem = gl.allocate_shared_memory(dtype, desc_q.block_type.shape, desc_q.layout)
    q_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(q_bar, count=1)
    mbarrier.expect(q_bar, desc_q.block_type.nbytes)
    tma.async_load(desc_q, [off_m, 0], q_bar, q_smem)
    mbarrier.wait(q_bar, phase=0)
    mbarrier.invalidate(q_bar)

    # Allocate circular SMEM buffers and mbarriers for K and V
    k_bufs = gl.allocate_shared_memory(dtype, [NUM_STAGES] + desc_k.block_type.shape, desc_k.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [NUM_STAGES] + desc_v.block_type.shape, desc_v.layout)
    k_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    k_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    v_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    v_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(NUM_STAGES):
        mbarrier.init(k_empty_bars.index(i), count=1)
        mbarrier.init(k_ready_bars.index(i), count=1)
        mbarrier.init(v_empty_bars.index(i), count=1)
        mbarrier.init(v_ready_bars.index(i), count=1)

    bufs = (k_bufs, v_bufs)
    bars = (k_empty_bars, k_ready_bars, v_empty_bars, v_ready_bars)

    # Default partition: 4-warp WGMMA consumer | Worker partition: 1-warp (24-reg) TMA producer
    gl.warp_specialize(
        [
            (fa3_compute_partition, (q_smem, desc_o, L, bufs, bars, off_m, N, num_kv_blocks,
                                     softmax_scale_log2, B_r, B_c, D, IS_CAUSAL)),
            (fa3_load_partition, (desc_k, desc_v, bufs, bars, num_kv_blocks, B_c)),
        ],
        [1],
        [24],
    )

    for i in gl.static_range(NUM_STAGES):
        mbarrier.invalidate(k_empty_bars.index(i))
        mbarrier.invalidate(k_ready_bars.index(i))
        mbarrier.invalidate(v_empty_bars.index(i))
        mbarrier.invalidate(v_ready_bars.index(i))


# ===-----------------------------------------------------------------------===#
# Host Entry Points & Benchmark Harness
# ===-----------------------------------------------------------------------===#


def flash_attention_3_forward(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = True,
        softmax_scale: Optional[float] = None,
        B_r: int = 64,
        B_c: int = 64,
        num_stages: int = 3,
):
    assert Q.is_cuda and Q.shape == K.shape == V.shape and Q.ndim == 2
    assert Q.dtype in (torch.float16, torch.bfloat16), "Hopper WGMMA requires float16 or bfloat16"
    assert num_stages >= 2 and B_r >= 64 and B_r % 64 == 0

    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    N, d = Q.shape
    D_PAD = triton.next_power_of_2(d)
    assert (d * Q.element_size()) % 16 == 0, "Head dim stride must be 16-byte aligned for TMA"

    softmax_scale_log2 = (1.0 / math.sqrt(d) if softmax_scale is None else softmax_scale) * LOG2_E
    O = torch.empty_like(Q)
    L = torch.empty((N,), dtype=torch.float32, device=Q.device)

    gl_dtype = gl.float16 if Q.dtype == torch.float16 else gl.bfloat16
    q_layout = gl.NVMMASharedLayout.get_default_for([B_r, D_PAD], gl_dtype)
    kv_layout = gl.NVMMASharedLayout.get_default_for([B_c, D_PAD], gl_dtype)

    desc_q = TensorDescriptor.from_tensor(Q, [B_r, D_PAD], q_layout)
    desc_k = TensorDescriptor.from_tensor(K, [B_c, D_PAD], kv_layout)
    desc_v = TensorDescriptor.from_tensor(V, [B_c, D_PAD], kv_layout)
    desc_o = TensorDescriptor.from_tensor(O, [B_r, D_PAD], q_layout)

    grid = (triton.cdiv(N, B_r),)
    flash_attention_3_gluon_kernel[grid](
        desc_q, desc_k, desc_v, desc_o, L, softmax_scale_log2,
        NUM_STAGES=num_stages, IS_CAUSAL=is_causal,
        num_warps=4, maxnreg=128,
    )
    return O, L


def flash_attention_3(N: int, d: int, dtype: torch.dtype = torch.float16, is_causal: bool = True):
    torch.manual_seed(0)
    Q = torch.randn((N, d), device=device, dtype=dtype)
    K = torch.randn((N, d), device=device, dtype=dtype)
    V = torch.randn((N, d), device=device, dtype=dtype)

    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    O, _ = flash_attention_3_forward(Q, K, V, is_causal=is_causal)
    torch.cuda.synchronize()
    return O, (time.perf_counter_ns() - start) / 10 ** 6


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    if not is_hopper():
        raise RuntimeError("FlashAttention-3 Gluon kernel requires an NVIDIA Hopper (SM90) GPU.")

    print("Using NVIDIA Hopper CUDA device...")
    d = 128
    dtype = torch.float16
    N_values = [2048 * (2 ** i) for i in range(8)]

    torch.manual_seed(0)
    Q = torch.randn((N_values[0], d), device=device, dtype=dtype)
    K = torch.randn((N_values[0], d), device=device, dtype=dtype)
    V = torch.randn((N_values[0], d), device=device, dtype=dtype)
    for causal in (False, True):
        actual, _ = flash_attention_3_forward(Q, K, V, is_causal=causal)
        expected = F.scaled_dot_product_attention(
            Q[None, None], K[None, None], V[None, None], is_causal=causal
        )[0, 0]
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        print(f"Correctness check passed (is_causal={causal})!")

    for N in N_values:
        print("Testing for N =", N)
        _ = flash_attention_3(N, d, dtype=dtype, is_causal=True)
        torch.cuda.synchronize()
        _, time_taken = flash_attention_3(N, d, dtype=dtype, is_causal=True)
        torch.cuda.synchronize()
        print(f"Execution time is {time_taken:.3f} ms")
        print("--------------------------------------------------------------------------")
