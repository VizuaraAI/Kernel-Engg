from numba import cuda
import cupy as cp
import numpy as np


"""
read from input_mat
write to output_mat

where to write in output_mat?
> out_row, out_col


"""


@cuda.jit
def tranpose(input_mat, output_mat):
    n, m = output_mat.shape
    
    out_row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  # type: ignore
    out_col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # type: ignore
    
    if out_col < m and out_row < n:
        # l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
        # l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
        # load/read operations are not coalesced
        # store/write operations are coalesced
        output_mat[out_row][out_col] = input_mat[out_col][out_row]

def main():
    m = 10000
    n = 2000
    
    device = cp.cuda.Device()
    
    input_mat = cp.random.uniform(-1, 1, (m, n), dtype=cp.float32) # type: ignore
    output_mat = cp.zeros((n, m), dtype=cp.float32)
    
    # launch configuration
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (m + threads_per_block[0] - 1) // threads_per_block[0],
        (n + threads_per_block[1] - 1) // threads_per_block[1],
    )
    tranpose[blocks_per_grid, threads_per_block](input_mat, output_mat) # type: ignore
    cuda.synchronize()
    
    input_mat_np = cp.asnumpy(input_mat)
    output_mat_np = cp.asnumpy(output_mat)
    np.testing.assert_allclose(input_mat_np.T, output_mat_np, rtol=1e-5, atol=1e-5)


main()