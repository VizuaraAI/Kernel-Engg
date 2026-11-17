import time

import torch
import triton
import triton.language as tl

from vanilla_attention import vanilla_attention


@triton.jit
def flash_attention_kernel(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        O: torch.Tensor,
        stride_qn, stride_qd,
        stride_kn, stride_kd,
        stride_vn, stride_vd,
        stride_on, stride_od,
        N: tl.constexpr,
        d: tl.constexpr,
        B_c: tl.constexpr, B_r: tl.constexpr
):
    pid = tl.program_id(axis=0)

    offset_qn = (pid * B_r) + tl.arange(0, B_r)
    offset_qd = tl.arange(0, triton.next_power_of_2(d))
    mask_q = (offset_qn < N)[:, None] & (offset_qd < d)[None, :]
    block_ptrs_q = Q + (
            (offset_qn * stride_qn)[:, None] + (offset_qd * stride_qd)[None, :]
    )
    Q_i = tl.load(
        pointer=block_ptrs_q,
        mask=mask_q,
        other=0.0
    )

    # 2. initialize O_i, l, m
    l_i = tl.zeros((B_r, 1), dtype=tl.float32)
    m_i = tl.full((B_r, 1), dtype=tl.float32, value=-float('inf'))
    O_i = tl.zeros((B_r, d), dtype=tl.float32)

    end_j = tl.minimum((pid + 1) * B_r, N)
    for j in range(0, end_j, B_c):
        offset_kn = j + tl.arange(0, B_c)
        offset_kd = tl.arange(0, triton.next_power_of_2(d))
        mask_k = (offset_kn[:, None] < N) & (offset_kd[None, :] < d)
        block_ptrs_k = K + (
                (offset_kn * stride_kn)[:, None] + (offset_kd * stride_kd)[None, :]
        )
        K_j = tl.load(
            pointer=block_ptrs_k,
            mask=mask_k,
            other=0.0
        )

        offset_vn = j + tl.arange(0, B_c)
        offset_vd = tl.arange(0, triton.next_power_of_2(d))
        mask_v = (offset_vn[:, None] < N) & (offset_vd[None, :] < d)
        block_ptrs_v = V + (
                (offset_vn * stride_vn)[:, None] + (offset_vd * stride_vd)[None, :]
        )
        V_j = tl.load(
            pointer=block_ptrs_v,
            mask=mask_v,
            other=0.0
        )

        S_ij = tl.dot(Q_i, tl.trans(K_j), allow_tf32=False)
        causal_mask = (offset_qn[:, None] >= offset_kn[None, :]) & (offset_kn[None, :] < N)
        S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        m_ij = tl.max(S_ij, axis=1, keep_dims=True)
        P_ij = tl.exp(S_ij - m_ij)
        l_ij = tl.sum(P_ij, axis=1, keep_dims=True)
        m_i_new = tl.maximum(m_i, m_ij)
        l_i_new = (tl.exp(m_i - m_i_new) * l_i) + (tl.exp(m_ij - m_i_new) * l_ij)

        O_i = (l_i * tl.exp(m_i - m_i_new) * O_i) + (tl.exp(m_ij - m_i_new) * tl.dot(P_ij, V_j, allow_tf32=False))
        O_i = (1.0 / l_i_new) * O_i

        l_i = l_i_new
        m_i = m_i_new

    offset_on = tl.arange(0, B_r) + (pid * B_r)
    offset_od = tl.arange(0, triton.next_power_of_2(d))
    mask_o = (offset_on < N)[:, None] & (offset_od < d)[None, :]
    block_ptrs_o = O + (
            (offset_on * stride_on)[:, None] + (offset_od * stride_od)[None, :]
    )
    tl.store(block_ptrs_o, value=O_i, mask=mask_o)


def flash_attention(N, d) -> torch.Tensor:
    torch.manual_seed(0)
    WQ = torch.randn(size=(d, d)).to(device)
    WK = torch.randn(size=(d, d)).to(device)
    WV = torch.randn(size=(d, d)).to(device)
    input_seq = torch.randn(size=(N, d)).to(device)
    B_c = 32
    B_r = 32
    Q = torch.matmul(input_seq, WQ)
    K = torch.matmul(input_seq, WK)
    V = torch.matmul(input_seq, WV)
    O = torch.zeros((N, d), dtype=Q.dtype, device=Q.device)
    grid = lambda meta: (triton.cdiv(N, B_r),)
    start = time.perf_counter_ns()
    flash_attention_kernel[grid](Q, K, V, O,
                                 Q.stride(0), Q.stride(1),
                                 K.stride(0), K.stride(1),
                                 V.stride(0), V.stride(1),
                                 O.stride(0), O.stride(1),
                                 N, d, B_c, B_r)
    end = time.perf_counter_ns()
    return O, (end - start) / 10 ** 6


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using the CUDA device...")

d = 128
N_values = [2048 * (2 ** i) for i in range(8)]

# assert if the kernel computes the correct outputs
actual, _ = flash_attention(N_values[0], d)
expected, _ = vanilla_attention(N_values[0], d)
torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

for N in N_values:
    print("Testing for N =", N)
    print("Compiling kernel (warm-up)...")
    _ = flash_attention(N, d)
    torch.cuda.synchronize()
    O, time_taken = flash_attention(N, d)
    torch.cuda.synchronize()
    print(f"Execution time is {time_taken:.3f} ms")
    print("--------------------------------------------------------------------------")
