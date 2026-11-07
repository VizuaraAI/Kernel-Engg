from numba import cuda
import cupy as cp
import numpy as np


"""
The my_naive_transpose.py had a massive slowdown when the matrix size was increased. 
This is due to uncoalesced memory access. We need to make sure that the threads
that are accessing the memory are accessing adjacent memory locations.
But there is a catch, if we coalesce the "load" function (i.e. reading from the input matrix), 
then we will have uncoalesced "store" access (i.e. writing to the output matrix)
How to fix this? 
Use a shared memory to store the loaded data and then write it to the output matrix.
This way, we can have coalesced access to both the input and output matrices.

"""

TILE_DIM = 32

@cuda.jit
def my_improved_transpose_kernel(input_matrix, output_matrix):
    n, m = output_matrix.shape

    tile = cuda.shared.array((TILE_DIM, TILE_DIM), dtype=cp.float32)

    # Why are blockIdx.y and blockIdx.x swapped here, while threadIdx.x and threadIdx.y are NOT?
    # 1. Grid dimensions are sized for output_matrix (n=2000 rows, m=10000 cols):
    #    - blockIdx.x spans m (0..312) 10000/32 -> corresponds to output columns, which are INPUT ROWS.
    #    - blockIdx.y spans n (0..62)  2000/32 -> corresponds to output rows, which are INPUT COLUMNS.
    #    So block (bx, by) reads the input tile at row-tile `bx` and col-tile `by`.
    # 2. Within that tile, threadIdx.x MUST stay paired with `input_col` (and threadIdx.y with `input_row`)
    #    because hardware warps group consecutive threadIdx.x (0..31). Pairing threadIdx.x with columns
    #    ensures all 32 threads in a warp read 32 consecutive floats along a row in HBM (coalesced load).
    # 3. TILE_DIM (32) equals both blockDim.x and blockDim.y because threads_per_block = (32, 32) is square.
    input_col = cuda.threadIdx.x + TILE_DIM * cuda.blockIdx.y
    input_row = cuda.threadIdx.y + TILE_DIM * cuda.blockIdx.x

    if input_row < m and input_col < n:
        # here is the confusing part but we clarify
        # the tiles index is given by the thread index, threadIdx.y and threadIdx.x
        # threadIdx.y moves in the rowwise and threadIdx.x moves in the columnwise
        # this has both load and store operations.
        # Load from global memory and store to shared memory
        # Access from global memory is coalesced
        # Access from shared memory is coalesced
        tile[cuda.threadIdx.y, cuda.threadIdx.x] = input_matrix[input_row, input_col]
    
    cuda.syncthreads()

    output_col = cuda.threadIdx.x + TILE_DIM * cuda.blockIdx.x
    output_row = cuda.threadIdx.y + TILE_DIM * cuda.blockIdx.y

    if output_row < n and output_col < m:
        # now copy back from the shared memory to the global memory
        # This also has load and store operations
        # load from the shared memory and store to global memory
        # access from shared memory is not coalesced but its fine, because the shared memory is on-chip
        # access from global memory is coalesced
        output_matrix[output_row, output_col] = tile[cuda.threadIdx.x, cuda.threadIdx.y]
    


def main():
    m = 10000
    n = 2000
    
    device = cp.cuda.Device()
    
    input_matrix = cp.random.uniform(-1, 1, size=(m, n), dtype=cp.float32)
    output_matrix = cp.zeros((n, m), dtype=cp.float32)

    # launch configuration
    threads_per_block = (TILE_DIM,TILE_DIM)
    
    blocks_per_grid = (
        (m + threads_per_block[0] -1) // threads_per_block[0],
        (n + threads_per_block[1] -1) // threads_per_block[1]
    )
    my_improved_transpose_kernel[blocks_per_grid, threads_per_block](input_matrix, output_matrix)
    cuda.synchronize()

    # copy from device to host
    input_matrix_host = cp.asnumpy(input_matrix)
    output_matrix_host = cp.asnumpy(output_matrix)

    # check if the output is correct
    np.testing.assert_allclose(output_matrix_host, input_matrix_host.T, rtol=1e-5, atol=1e-5 )
    print("test passed!")

    
    
    


if __name__ == '__main__':
    main()