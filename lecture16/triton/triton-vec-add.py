import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def vector_add(
        x_ptr,
        y_ptr,
        sum_ptr,
        n,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(sum_ptr + offsets, output, mask=mask)

def main():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    sum = torch.empty_like(x)

    n_elements = sum.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    vector_add[grid](x, y, sum, n_elements, BLOCK_SIZE=1024)

    print(torch.allclose(sum, x + y, rtol=1e-2, atol=1e-2))

main()