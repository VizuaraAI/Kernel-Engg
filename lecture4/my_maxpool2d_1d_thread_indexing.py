from numba import cuda
import cupy as cp

@cuda.jit
def maxpool_1d(input_mat, output_mat, pool_size):
    
    globalThreadIdx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    out_n = output_mat.shape[0]
    
    out_row = globalThreadIdx % out_n
    out_col = globalThreadIdx // out_n

    if out_row >= out_n or out_col >= out_n:
        return
    
    input_row = out_row * pool_size
    input_col = out_col * pool_size

    max_val = input_mat[input_row, input_col]

    for i in range(pool_size):
        for j in range(pool_size):
            max_val = max(
                max_val,
                input_mat[input_row + i, input_col + j]
            )
    output_mat[out_row, out_col] = max_val
    
    


def main():
    n = 10000
    pool_size = 4
    output_n = n // pool_size

    device = cp.cuda.Device()

    input_mat = cp.random.uniform(0,1,(n,n), dtype=cp.float32)
    output_mat = cp.zeros((output_n, output_n), dtype=cp.float32)

    # GPU activation
    num_threads_per_block = (1024, 1)
    num_blocks_per_grid = (
        int(cp.ceil((output_n * output_n) / num_threads_per_block[0])),
        1
        )

    maxpool_1d[num_blocks_per_grid, num_threads_per_block](input_mat, output_mat, pool_size)

    cuda.synchronize()
    cpu_input = cp.asnumpy(input_mat)
    cpu_output = cp.asnumpy(output_mat)

    print("GPU Matrix (first 10x10):")
    print(cpu_input[:10, :10])
    print("\nGPU Output (first 5x5):")
    print(cpu_output[:5, :5])


    


    





if __name__ == "__main__":
    main()