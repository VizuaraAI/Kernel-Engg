from numba import cuda
import cupy as cp
import numpy as np


"""
Read from input matrix and write to output matrix

Where to write in output matrix: (row, col)
out_row, out_col

"""
@cuda.jit
def my_naive_transpose_kernel(input_matrix, output_matrix):
    n, m = output_matrix.shape

    out_row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    out_col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if out_row < n and out_col <m:
        output_matrix[out_row, out_col] = input_matrix[out_col, out_row]



def main():
    m = 10000
    n = 2000
    device = cp.cuda.Device()

    input_matrix = cp.random.uniform(-1, 1, (m, n), dtype=cp.float32)
    output_matrix = cp.zeros((n, m), dtype=cp.float32)

    # launch configuration

    threads_per_block = (32, 32)
    blocks_per_grid = (
        (m + threads_per_block[0] - 1 ) // threads_per_block[0],
        (n + threads_per_block[1] - 1) // threads_per_block[1]
    )

    my_naive_transpose_kernel[blocks_per_grid, threads_per_block](input_matrix, output_matrix)

    cuda.synchronize()

    input_matrix_host = cp.asnumpy(input_matrix)
    output_matrix_host = cp.asnumpy(output_matrix)

    np.testing.assert_allclose(input_matrix_host.T, output_matrix_host, rtol=1e-5, atol=1e-5)

    print("Test passed")


    


if __name__ == "__main__":
    main()