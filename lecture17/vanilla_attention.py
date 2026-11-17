import time

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using the CUDA device...")


def vanilla_attention(
        N,  # input sequence length
        d,  # input sequence token dimensionality
) -> torch.Tensor:
    torch.manual_seed(0)
    WQ = torch.randn(size=(d, d)).to(device)
    WK = torch.randn(size=(d, d)).to(device)
    WV = torch.randn(size=(d, d)).to(device)
    mask = torch.triu(torch.ones((N, N), device=device, dtype=torch.bool), diagonal=1)
    input_seq = torch.randn(size=(N, d)).to(device)
    Q = torch.matmul(input_seq, WQ)  # dims: (N, d)
    K = torch.matmul(input_seq, WK)  # dims: (N, d)
    V = torch.matmul(input_seq, WV)  # dims: (N, d)

    start = time.perf_counter_ns()
    attention = torch.matmul(Q, K.T)  # dims: (N, N)
    attention = attention.masked_fill(mask, float("-inf"))
    attn_weights = F.softmax(attention, dim=-1)
    O = torch.matmul(attn_weights, V)  # dims: (N, d)
    end = time.perf_counter_ns()
    return O, (end - start) / 10 ** 6


def main():
    N_values = [2048 * (2 ** i) for i in range(8)]
    d = 128
    for N in N_values:
        print("Testing for N =", N)
        print("Compiling kernel (warm-up)...")
        _ = vanilla_attention(N, d)
        torch.cuda.synchronize()
        _, time_taken = vanilla_attention(N, d)
        torch.cuda.synchronize()
        print(f"Execution time is {time_taken:.3f} ms")
        print("--------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
