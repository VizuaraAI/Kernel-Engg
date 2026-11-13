import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def vector_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    # Thread/block indices
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    m, n = gA.shape

    # Map the linear index to a 2D coordinate
    mi = idx // n
    ni = idx % n

    if mi < m:
        gC[mi, ni] = gA[mi, ni] + gB[mi, ni]


@cute.jit
def vector_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    num_threads = 256
    m, n = mA.shape
    total = m * n

    vector_add_kernel(mA, mB, mC).launch(
        grid=((total + num_threads - 1) // num_threads, 1, 1),
        block=(num_threads, 1, 1),
    )


if __name__ == "__main__":
    M, N = 1024, 1024
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.randn(M, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    # Wrap PyTorch tensors as CuTe tensors via DLPack (zero-copy)
    a_ = from_dlpack(a)
    b_ = from_dlpack(b)
    c_ = from_dlpack(c)

    # Compile once, reuse many times
    compiled = cute.compile(vector_add, a_, b_, c_)
    compiled(a_, b_, c_)

    torch.testing.assert_close(c, a + b)
    print("Success!")