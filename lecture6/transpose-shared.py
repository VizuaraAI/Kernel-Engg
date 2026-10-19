import numba
from numba import cuda
import cupy as cp
import numpy as np

TILE_DIM = 32
BLOCK_ROWS = 32
TILE_DIM_PADDED = 33

@cuda.jit
def transpose(input_mat, output_mat):
    # Use the pre-calculated constant so Numba sees a pure Tuple of literals
    tile = cuda.shared.array((TILE_DIM, TILE_DIM_PADDED), dtype=numba.float32)
    
    m = input_mat.shape[0]
    n = input_mat.shape[1]

    in_col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    in_row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if in_row < m and in_col < n:
        tile[cuda.threadIdx.y, cuda.threadIdx.x] = input_mat[in_row, in_col]

    cuda.syncthreads()

    out_col = cuda.threadIdx.x + cuda.blockIdx.y * TILE_DIM   
    out_row = cuda.threadIdx.y + cuda.blockIdx.x * TILE_DIM   
    
    if out_row < n and out_col < m:
        output_mat[out_row, out_col] = tile[cuda.threadIdx.x, cuda.threadIdx.y]

def main():
    m = 10000
    n = 2000
    
    device = cp.cuda.Device()
    
    input_mat = cp.random.uniform(-1, 1, (m, n), dtype=cp.float32)
    output_mat = cp.zeros((n, m), dtype=cp.float32)
    
    threads_per_block = (TILE_DIM, BLOCK_ROWS)
    blocks_per_grid = (
        (n + TILE_DIM - 1) // TILE_DIM,
        (m + TILE_DIM - 1) // TILE_DIM,
    )
    
    transpose[blocks_per_grid, threads_per_block](input_mat, output_mat)
    device.synchronize()
    
    input_mat_np = cp.asnumpy(input_mat)
    output_mat_np = cp.asnumpy(output_mat)
    
    assert input_mat_np.shape[0] == output_mat_np.shape[1]
    assert input_mat_np.shape[1] == output_mat_np.shape[0]
    
    np.testing.assert_allclose(input_mat_np.T, output_mat_np, rtol=1e-5, atol=1e-5)
    print("Transpose successful and verified.")

if __name__ == "__main__":
    main()