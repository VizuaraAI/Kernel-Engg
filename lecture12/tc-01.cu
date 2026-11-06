#include "../common/cuda_utils.cuh"
#include <cuda_fp16.h>
#include <iostream>
#include <mma.h>

using namespace nvcuda;

// Tensor Cores operate on specific tile sizes. 16x16x16 is standard for FP16.
const int M = 16;
const int N = 16;
const int K = 16;

__global__ void wmmaSingleWarpKernel(const half* a, const half* b, float* c) {
    // 1. Declare WMMA fragments (registers dedicated to Tensor Core operations)
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> fragmentA;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> fragmentB;
    wmma::fragment<wmma::accumulator, M, N, K, float> fragmentC;

    // 2. Initialize the accumulator fragment to 0
    wmma::fill_fragment(fragmentC, 0.0f);

    // 3. Load data from global memory into the fragments
    // The last argument is the leading dimension (stride) of the matrix
    wmma::load_matrix_sync(fragmentA, a, K);
    wmma::load_matrix_sync(fragmentB, b, K);

    // 4. Perform the matrix multiply-accumulate (D = A * B + C)
    wmma::mma_sync(fragmentC, fragmentA, fragmentB, fragmentC);

    // 5. Store the result from the fragment back to global memory
    wmma::store_matrix_sync(c, fragmentC, N, wmma::mem_row_major);
}

int main() {
    half *hostMatA, *hostMatB;
    float* hostMatC;
    half *deviceMatA, *deviceMatB;
    float* deviceMatC;
    initializeMatrix<half>(hostMatA, deviceMatA, M, K, __float2half(2.0f));
    initializeMatrix<half>(hostMatB, deviceMatB, K, N, __float2half(3.0f));
    initializeMatrix<float>(hostMatC, deviceMatC, M, N, __float2half(0.0f));

    // Launch configuration: 1 block, 32 threads (exactly 1 warp)
    dim3 gridDim(1);
    dim3 blockDim(32);

    wmmaSingleWarpKernel<<<gridDim, blockDim>>>(deviceMatA, deviceMatB, deviceMatC);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(hostMatC, deviceMatC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify output (16 * 2.0 * 3.0 = 96.0)
    std::cout << "Top-left element result: " << hostMatC[0] << std::endl;
    if (hostMatC[0] == 96.0f) {
        std::cout << "Success! Tensor Core matmul executed correctly." << std::endl;
    }

    freeMatrix<half>(hostMatA, deviceMatA);
    freeMatrix<half>(hostMatB, deviceMatB);
    freeMatrix<float>(hostMatC, deviceMatC);

    return 0;
}