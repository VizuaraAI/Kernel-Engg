from numba import cuda
import cupy as cp
import numpy as np

@cuda.jit
def transpose(input_mat, output_mat):
    m = input_mat.shape[0]
    n = input_mat.shape[1]
    
    out_col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x # type: ignore
    out_row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y # type: ignore
    
    if out_col < n and out_row < m:
        output_mat[out_col][out_row] = input_mat[out_row][out_col]

def main():
    m = 10000
    n = 2000
    
    device = cp.cuda.Device()
    
    input_mat = cp.random.uniform(-1, 1, (m, n), dtype=cp.float32) # type: ignore
    output_mat = cp.zeros((n, m), dtype=cp.float32)
    
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (n + threads_per_block[0] - 1) // threads_per_block[0],
        (m + threads_per_block[1] - 1) // threads_per_block[1]
    )
    transpose[blocks_per_grid, threads_per_block](input_mat, output_mat) # type: ignore
    device.synchronize()
    
    input_mat_np = cp.asnumpy(input_mat)
    output_mat_np = cp.asnumpy(output_mat)
    
    assert input_mat_np.shape[0] == output_mat_np.shape[1]
    assert input_mat_np.shape[1] == output_mat_np.shape[0]
    np.testing.assert_allclose(input_mat_np.T, output_mat_np, rtol=1e-5, atol=1e-5)
    print("Transpose successful and verified.")


if __name__ == "__main__":
    main()