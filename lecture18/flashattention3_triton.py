"""
FlashAttention-3 forward pass in Triton (similar to lecture17/flash_attention_triton.py).

Implements the core FlashAttention-3 techniques from:
  "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
  (Shah et al., 2024)

Key differences from FlashAttention-1 (lecture17/flash_attention_triton.py):
  1. Single-launch query-parallel execution (Algorithm 1):
     Parallelizes over query blocks Q_i (grid = (T_r,)) and keeps Q_i, O_i, l_i, m_i
     on-chip across all K_j, V_j iterations instead of reloading/storing them per block.
  2. Hopper TMA loads/stores:
     Uses `tl.make_tensor_descriptor` (`desc.load` / `desc.store`) for asynchronous
     HBM <-> SMEM tile transfers (requires `triton.set_allocator` and SM90+).
  3. Producer-Consumer Warp Specialization + Circular SMEM Buffer (§3.1):
     Uses `tl.range(..., num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE)` so
     the Triton compiler automatically partitions TMA-producer and compute-consumer
     warps with multi-stage pipelining and dynamic register reallocation.
  4. Two-phase causal loop + log2-domain online softmax (`MUFU.EX2`):
     Splits the inner loop into unmasked off-diagonal blocks (Stage 1) and causally
     masked diagonal blocks (Stage 2), folding log2(e) into the scores.
"""

import time
from typing import Optional

import torch
import triton
import triton.language as tl

from vanilla_attention import vanilla_attention

LOG2_E = tl.constexpr(1.4426950408889634)
LN_2 = tl.constexpr(0.6931471805599453)


def _alloc_fn(size: int, alignment: int, stream: Optional[int]):
    """Workspace allocator required by Triton for device-side TMA descriptors."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def _attn_fwd_inner(
        O_i,
        l_i,
        m_i,
        Q_i,
        desc_k,
        desc_v,
        pid,
        offset_m,
        offset_n_base,
        N: tl.constexpr,
        B_c: tl.constexpr,
        B_r: tl.constexpr,
        STAGE: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        WARP_SPECIALIZE: tl.constexpr,
):
    # STAGE 1: strictly off-diagonal blocks (j < i * B_r) -> fully causal, no mask needed
    # STAGE 2: diagonal block(s) (i * B_r <= j < (i + 1) * B_r) -> apply causal + tail mask
    if STAGE == 1:
        lo = 0
        hi = pid * B_r
    else:
        lo = pid * B_r
        hi = tl.minimum((pid + 1) * B_r, N)

    # Producer-consumer warp specialization with an s-stage circular SMEM buffer (Alg. 1)
    for start_j in tl.range(
            lo,
            hi,
            B_c,
            num_stages=NUM_STAGES,
            warp_specialize=WARP_SPECIALIZE,
    ):
        start_j = tl.multiple_of(start_j, B_c)

        # Producer warps asynchronously load K_j via TMA into circular SMEM buffer
        K_j = desc_k.load([start_j, 0])

        # Consumer warps compute S_ij = Q_i K_j^T (GEMM0)
        S_ij = tl.dot(Q_i, tl.trans(K_j), allow_tf32=False) * LOG2_E

        if STAGE == 2:
            offset_n = start_j + offset_n_base
            causal_mask = (offset_m[:, None] >= offset_n[None, :]) & (offset_n[None, :] < N)
            S_ij = tl.where(causal_mask, S_ij, -float("inf"))

        # Online softmax in log2 domain (maps directly to Hopper MUFU.EX2)
        m_ij = tl.max(S_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        P_ij = tl.math.exp2(S_ij - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(P_ij, axis=1)

        # Rescale accumulator O_i and accumulate P_ij V_j (GEMM1)
        O_i = O_i * alpha[:, None]
        V_j = desc_v.load([start_j, 0])
        O_i = tl.dot(P_ij.to(V_j.dtype), V_j, acc=O_i, allow_tf32=False)
        m_i = m_i_new

    return O_i, l_i, m_i


@triton.jit
def flash_attention_3_kernel(
        Q,
        K,
        V,
        O,
        L,
        stride_qn, stride_qd,
        stride_kn, stride_kd,
        stride_vn, stride_vd,
        stride_on, stride_od,
        stride_l,
        N: tl.constexpr,
        d: tl.constexpr,
        D_PAD: tl.constexpr,
        B_c: tl.constexpr,
        B_r: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        WARP_SPECIALIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Device-side TMA tensor descriptors for Q, K, V, O (2D tiles of shape [B, D_PAD])
    desc_q = tl.make_tensor_descriptor(
        Q,
        shape=[N, d],
        strides=[stride_qn, stride_qd],
        block_shape=[B_r, D_PAD],
    )
    desc_k = tl.make_tensor_descriptor(
        K,
        shape=[N, d],
        strides=[stride_kn, stride_kd],
        block_shape=[B_c, D_PAD],
    )
    desc_v = tl.make_tensor_descriptor(
        V,
        shape=[N, d],
        strides=[stride_vn, stride_vd],
        block_shape=[B_c, D_PAD],
    )
    desc_o = tl.make_tensor_descriptor(
        O,
        shape=[N, d],
        strides=[stride_on, stride_od],
        block_shape=[B_r, D_PAD],
    )

    offset_m = pid * B_r + tl.arange(0, B_r)
    offset_n_base = tl.arange(0, B_c)

    # On-chip initialization of O_i, l_i, m_i (Algorithm 1, line 13)
    O_i = tl.zeros([B_r, D_PAD], dtype=tl.float32)
    l_i = tl.zeros([B_r], dtype=tl.float32)
    m_i = tl.full([B_r], -float("inf"), dtype=tl.float32)

    # Load Q_i once from HBM to on-chip SRAM via TMA and reuse across all K_j, V_j blocks
    Q_i = desc_q.load([pid * B_r, 0])

    # Stage 1: off-diagonal causal blocks (no masking required)
    O_i, l_i, m_i = _attn_fwd_inner(
        O_i, l_i, m_i, Q_i,
        desc_k, desc_v,
        pid, offset_m, offset_n_base,
        N, B_c, B_r,
        STAGE=1,
        NUM_STAGES=NUM_STAGES,
        WARP_SPECIALIZE=WARP_SPECIALIZE,
    )

    # Stage 2: diagonal causal block(s) (with causal mask)
    O_i, l_i, m_i = _attn_fwd_inner(
        O_i, l_i, m_i, Q_i,
        desc_k, desc_v,
        pid, offset_m, offset_n_base,
        N, B_c, B_r,
        STAGE=2,
        NUM_STAGES=NUM_STAGES,
        WARP_SPECIALIZE=WARP_SPECIALIZE,
    )

    # Epilogue: O_i = diag(l_i)^-1 O_i, L_i = m_i + log(l_i) (Algorithm 1, lines 24-25)
    O_i = O_i / l_i[:, None]
    L_i = (m_i + tl.math.log2(l_i)) * LN_2

    # TMA store automatically clips out-of-bounds rows past N
    desc_o.store([pid * B_r, 0], O_i.to(O.dtype.element_ty))
    tl.store(L + offset_m * stride_l, L_i, mask=offset_m < N)


def flash_attention_3(N, d, warp_specialize=True):
    torch.manual_seed(0)
    WQ = torch.randn(size=(d, d)).to(device)
    WK = torch.randn(size=(d, d)).to(device)
    WV = torch.randn(size=(d, d)).to(device)
    input_seq = torch.randn(size=(N, d)).to(device)

    Q = torch.matmul(input_seq, WQ)
    K = torch.matmul(input_seq, WK)
    V = torch.matmul(input_seq, WV)

    B_r = 64
    B_c = 64
    D_PAD = triton.next_power_of_2(d)
    NUM_STAGES = 2

    O = torch.empty_like(Q)
    L = torch.empty((N,), dtype=torch.float32, device=Q.device)

    triton.set_allocator(_alloc_fn)

    T_r = triton.cdiv(N, B_r)
    grid = (T_r,)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    flash_attention_3_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        L.stride(0),
        N=N,
        d=d,
        D_PAD=D_PAD,
        B_c=B_c,
        B_r=B_r,
        NUM_STAGES=NUM_STAGES,
        WARP_SPECIALIZE=warp_specialize,
        num_warps=4,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter_ns()
    return O, (end - start) / 10 ** 6


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using the CUDA device...")

    d = 128
    N_values = [2048 * (2 ** i) for i in range(8)]

    # Assert that the kernel computes the correct outputs against vanilla_attention
    actual, _ = flash_attention_3(N_values[0], d)
    expected, _ = vanilla_attention(N_values[0], d)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    print("Correctness check passed!")

    for N in N_values:
        print("Testing for N =", N)
        print("Compiling kernel (warm-up)...")
        _ = flash_attention_3(N, d)
        torch.cuda.synchronize()
        O, time_taken = flash_attention_3(N, d)
        torch.cuda.synchronize()
        print(f"Execution time is {time_taken:.3f} ms")
        print("--------------------------------------------------------------------------")
