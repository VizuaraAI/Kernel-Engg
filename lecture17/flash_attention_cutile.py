import time

import cuda.tile as ct
import torch

from vanilla_attention import vanilla_attention


@ct.kernel
def flash_attention_kernel(
        Q: ct.IndexedWithInt64,
        K: ct.IndexedWithInt64,
        V: ct.IndexedWithInt64,
        O: ct.IndexedWithInt64,
        N: ct.Constant[int],
        d: ct.Constant[int],
        B_c: ct.Constant[int], B_r: ct.Constant[int]
):
    bid = ct.bid(axis=0)
    # load the block from Q (queries) of shape (B_r, d)
    Q_i = ct.load(
        Q,
        index=(bid, 0),
        shape=(B_r, d)
    )

    # running statistics used for computation of softmax:
    # (1) running sum of exp sums (row-wise)
    l_i = ct.zeros((B_r, 1), dtype=ct.float32)
    # (2) max of elements (row-wise)
    m_i = ct.full((B_r, 1), dtype=ct.float32, fill_value=-float('inf'))

    # output matrix
    O_i = ct.zeros((B_r, d), dtype=ct.float32)

    num_tiles_along_c = ct.num_tiles(K, shape=(B_c, d), axis=0)
    end_tile = min(ct.cdiv((bid + 1) * B_r, B_c), num_tiles_along_c)
    # if B_r = B_c, end_tile = bid + 1
    # hence 0 <= j < bid
    for j in range(end_tile):
        K_j = ct.load(
            K,
            index=(j, 0),
            shape=(B_c, d)
        )

        V_j = ct.load(
            V,
            index=(j, 0),
            shape=(B_c, d)
        )

        # Note: any operation along axis=1 is rowwise
        S_ij = ct.matmul(Q_i, ct.transpose(K_j))
        col_idx = j * B_c + ct.arange(B_c, dtype=ct.int32)[None, :]
        row_idx = bid * B_r + ct.arange(B_r, dtype=ct.int32)[:, None]  # shape: (B_r, 1)
        causal_mask = (row_idx >= col_idx) & (col_idx < N)
        S_ij = ct.where(causal_mask, S_ij, -float('inf'))
        m_ij = ct.max(S_ij, axis=1, keepdims=True)
        P_ij = ct.exp(S_ij - m_ij)
        l_ij = ct.sum(P_ij, axis=1, keepdims=True)
        m_i_new = ct.maximum(m_i, m_ij)
        l_i_new = (ct.exp(m_i - m_i_new) * l_i) + (ct.exp(m_ij - m_i_new) * l_ij)

        O_i = (l_i * ct.exp(m_i - m_i_new) * O_i) + (ct.exp(m_ij - m_i_new) * ct.matmul(P_ij, V_j))
        O_i = (1.0 / l_i_new) * O_i

        l_i = l_i_new
        m_i = m_i_new

    ct.store(
        O,
        index=(bid, 0),
        tile=O_i
    )


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
    grid = (
        ct.cdiv(N, B_r),  # ct.bid(axis=0)
        1,
        1
    )
    start = time.perf_counter_ns()
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        flash_attention_kernel,
        (Q, K, V, O, N, d, B_c, B_r)
    )
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
    _, time_taken = flash_attention(N, d)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    print(f"Execution time is {time_taken:.3f} ms")
    print("--------------------------------------------------------------------------")
