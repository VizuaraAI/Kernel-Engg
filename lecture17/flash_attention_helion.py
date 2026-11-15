import time

import helion
import helion.language as hl
import torch

from vanilla_attention import vanilla_attention


@helion.kernel(
    settings=helion.Settings(
        dot_precision="ieee",  # use IEEE float32 spec, instead of Nvidia's TF32
        autotune_effort="none",  # disable auto-tuning completely
        ignore_warnings=[
            helion.exc.FastMathUnused
        ]
    )
)
def flash_attention_kernel(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
) -> torch.Tensor:
    N, d = Q.shape
    d = hl.specialize(d)
    O = torch.empty_like(Q)

    for tile_q in hl.tile(N):
        Q_i = Q[tile_q, :]
        m_i = hl.full([tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([tile_q], dtype=torch.float32)
        O_i = hl.zeros([tile_q, d], dtype=torch.float32)

        for tile_k in hl.tile(0, tile_q.end):
            K_j = K[tile_k, :]
            V_j = V[tile_k, :]

            S_ij = hl.dot(Q_i, K_j.T)
            causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
            S_ij = torch.where(causal_mask, S_ij, float("-inf"))

            m_ij = torch.amax(S_ij, dim=-1)
            m_i_new = torch.maximum(m_i, m_ij)
            P_ij = torch.exp(S_ij - m_i_new[:, None])
            l_ij = torch.sum(P_ij, dim=-1)

            alpha = torch.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            O_i = alpha[:, None] * l_i[:, None] * O_i + hl.dot(P_ij.to(V_j.dtype), V_j)
            O_i = O_i / l_i_new[:, None]

            l_i = l_i_new
            m_i = m_i_new

        O[tile_q, :] = O_i.to(O.dtype)

    return O


def flash_attention(N: int, d: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(0)
    WQ = torch.randn(size=(d, d)).to(device)
    WK = torch.randn(size=(d, d)).to(device)
    WV = torch.randn(size=(d, d)).to(device)
    input_seq = torch.randn(size=(N, d)).to(device)

    Q = torch.matmul(input_seq, WQ)
    K = torch.matmul(input_seq, WK)
    V = torch.matmul(input_seq, WV)
    return flash_attention_kernel(Q, K, V)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = 128
N_values = [2048 * (2 ** i) for i in range(8)]

actual = flash_attention(N_values[0], d, device)
torch.manual_seed(0)
expected = vanilla_attention(N_values[0], d)
torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

for N in N_values:
    print("Testing for N =", N)
    _ = flash_attention(N, d, device)  # warm-up / autotune

    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    O = flash_attention(N, d, device)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()

    print(f"Execution time is {(end - start) / 10 ** 6:.3f} ms")
