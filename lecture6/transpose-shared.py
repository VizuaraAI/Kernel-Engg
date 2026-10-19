from numba import cuda
import cupy as cp
import numpy as np

TILE_DIM = 32
TILE_DIM_ONE = TILE_DIM + 1

@cuda.jit
def transpose(input_mat, output_mat):
    n, m = output_mat.shape
    tile = cuda.shared.array((TILE_DIM, TILE_DIM_ONE), dtype=cp.float32)
    
    # Phase 1
    # copy from input_mat (gmem) to tile (smem)
    in_col = cuda.threadIdx.x + cuda.blockIdx.y * cuda.blockDim.x # type: ignore
    in_row = cuda.threadIdx.y + cuda.blockIdx.x * cuda.blockDim.y # type: ignore
    if in_row < m and in_col < n:
        tile[cuda.threadIdx.y, cuda.threadIdx.x] = input_mat[in_row, in_col]
        
    cuda.syncthreads()
    
    # Phase 2
    # copy from tile (smem) to output_mat (gmem)
    out_col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x # type: ignore
    out_row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y # type: ignore
    if out_row < n and out_col < m:
        output_mat[out_row, out_col] = tile[cuda.threadIdx.x, cuda.threadIdx.y]


def main():
    m = 10000
    n = 1200
    
    input_mat = cp.random.uniform(-1, 1, (m, n), dtype=cp.float32)
    output_mat = cp.zeros((n, m), dtype=cp.float32)
    
    # launch configuration
    threads_per_block = (TILE_DIM, TILE_DIM)
    blocks_per_grid = (
        (m + threads_per_block[0] - 1) // threads_per_block[0],
        (n + threads_per_block[1] - 1) // threads_per_block[1],
    )
    transpose[blocks_per_grid, threads_per_block](input_mat, output_mat)
    cuda.synchronize()
    
    input_mat_np = cp.asnumpy(input_mat)
    output_mat_np = cp.asnumpy(output_mat)
    np.testing.assert_allclose(input_mat_np.T, output_mat_np, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    main()