import torch
import helion
import helion.language as hl
from lecture15.utils import benchmark_kernel

@helion.kernel(config = helion.Config(block_sizes = [32,]))
def vec_add_kernel_manual_tuning(
    x: torch.Tensor,
    y: torch.Tensor
):
    n = x.size(0)
    out = torch.zeros_like(x)
    for tile in hl.tile(n):
        out[tile] = x[tile] + y[tile]
    return out

def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    size = 1024 * 1024
    x = torch.rand(size=(size, ), dtype=torch.float32, device=device)
    y = torch.rand(size=(size, ), dtype=torch.float32, device=device)
    sum = vec_add_kernel_manual_tuning(x, y)
    torch.testing.assert_close(sum, x + y)

    benchmark_kernel(vec_add_kernel_manual_tuning, x, y)

main()