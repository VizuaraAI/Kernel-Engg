from numba import cuda
import cupy as cp

@cuda.jit
def maxpool2d(input_mat, output_mat, poolsize):
    out_col = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    out_row = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    
    out_n = output_mat.shape[0]
    
    if out_col >= out_n or out_row >= out_n:
        return
    
    # input mat coordinates
    in_row_start = out_row * poolsize
    in_col_start = out_col * poolsize
    
    # in_max_val is the maximum value observed 
    # within the pool window/tile/section of input_mat
    in_max_val = input_mat[in_row_start, in_col_start]
    for i in range(poolsize):
        for j in range(poolsize):
            in_max_val = max(
                in_max_val,
                input_mat[in_row_start + i, in_col_start + j]
            )  
    output_mat[out_row, out_col] = in_max_val
        

def main():
    # host buffers (will hold the source matrix)
    
    n = 4
    pool_size = 4
    output_n = n // pool_size
    
    device = cp.cuda.Device()
    
    input_mat = cp.random.uniform(0, 1, (n, n), dtype=cp.float32)
    output_mat = cp.zeros(shape=(output_n, output_n), dtype=cp.float32)
    
    # launch configuration
    threads_per_block = (32, 32)
    blocks_per_grid = (
        output_n // threads_per_block[0] + 1,
        output_n // threads_per_block[1] + 1
    )
    
    maxpool2d[blocks_per_grid, threads_per_block](input_mat, output_mat, pool_size)
    device.synchronize()
    
    input_mat_np = cp.asnumpy(input_mat)
    output_mat_np = cp.asnumpy(output_mat)
    
    print(input_mat_np)
    print(output_mat_np)
    print(output_mat_np.shape)
    import numpy as np
    expected = input_mat_np.reshape(output_n, pool_size, output_n, pool_size).max(axis=(1, 3))
    np.testing.assert_allclose(output_mat_np, expected)
    print("PASS: GPU output matches NumPy max pooling")
    
    
    

main()