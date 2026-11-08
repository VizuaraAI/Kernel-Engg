"""
Taken from https://helionlang.com/helion_tutorials.html#id1

This example demonstrates writing a matmul kernel using Helion
and Helion's auto-tuning capabilities
"""
import torch
import helion
import helion.language as hl
import time

# Define a matrix multiplication kernel
@helion.kernel()  # No config means auto-tuning will be used
def matmul_autotune(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out

# Create larger tensors for better auto-tuning results
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# First run will trigger auto-tuning
print("Running with auto-tuning (this might take a while)...")
start = time.time()
result = matmul_autotune(x, y)
end = time.time()
print(f"First run time (including auto-tuning): {end - start:.2f}s")

# Second run will use the tuned configuration
start = time.time()
result = matmul_autotune(x, y)
end = time.time()
print(f"Second run time (using tuned config): {end - start:.2f}s")

# Verify correctness
expected = x @ y
print(f"Result is correct: {torch.allclose(result, expected, rtol=1e-2, atol=1e-2)}")