import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def gemm(
        a_ptr, b_ptr, c_ptr,
        m, n, k,
        BLOCK_SIZE: tl.constexpr
):
    pidx = tl.program_id(axis=0)
    pidy = tl.program_id(axis=1)

    offsets_x = pidx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_y = pidy * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    c_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for i in range(0, n, BLOCK_SIZE):
        n_offsets = i + tl.arange(0, BLOCK_SIZE)

        a_block = tl.load(
            a_ptr + offsets_x[:, None] * n + n_offsets[None, :],
            mask=(offsets_x[:, None] < m) & (n_offsets[None, :] < n),
            other=0.0
        )
        b_block = tl.load(
            b_ptr + n_offsets[:, None] * k + offsets_y[None, :],
            mask=(n_offsets[:, None] < n) & (offsets_y[None, :] < k),
            other=0.0
        )
        c_block += tl.dot(a_block, b_block)

    tl.store(
        c_ptr + offsets_x[:, None] * k + offsets_y[None, :],
        c_block,
        mask=(offsets_x[:, None] < m) & (offsets_y[None, :] < k)
    )


def main():
    torch.manual_seed(0)
    m = 1024
    n = 800
    k = 1000
    BLOCK_SIZE = 64
    x = torch.rand((m, n), device=DEVICE)
    y = torch.rand((n, k), device=DEVICE)
    prod = torch.zeros((m, k), device=DEVICE)

    grid = (
        triton.cdiv(m, BLOCK_SIZE),
        triton.cdiv(k, BLOCK_SIZE),
    )
    gemm[grid](x, y, prod, m, n, k, BLOCK_SIZE=BLOCK_SIZE)

    diff = (prod - x @ y).abs()
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())
    print(torch.allclose(prod, x @ y, atol=1e-2, rtol=1e-2))


main()
