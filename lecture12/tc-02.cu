#include "../common/cuda_utils.cuh"
#include <cuda_fp16.h>
#include <iostream>
#include <mma.h>

using namespace nvcuda;

// Tensor Core tile dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel expects PADDED dimensions (guaranteed multiples of 16)
__global__ void wmmaMatmulAnySize(const half* matA, const half* matB, float* matC, const int M, const int N, const int K) {
    // 1. Map threads to 16x16 output tiles
    // blockDim.x = 32 (1 warp). blockDim.y = warps per block.
    const int warpM = blockIdx.y * blockDim.y + threadIdx.y; // Tile Row
    const int warpN = blockIdx.x;                            // Tile Col

    const int row = warpM * WMMA_M;
    const int col = warpN * WMMA_N;

    // Exit if warp is mapped completely outside the padded matrix
    if (row >= M || col >= N)
        return;

    // 2. Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // 3. Loop over the K dimension, sliding the 16x16 window
    for (int k = 0; k < K; k += WMMA_K) {
        // Load tiles. The trailing argument is the leading dimension (stride) of the padded matrix.
        wmma::load_matrix_sync(a_frag, matA + (row * K) + k, K);
        wmma::load_matrix_sync(b_frag, matB + (k * N) + col, N);

        // Accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4. Store the resulting tile back to global memory
    wmma::store_matrix_sync(matC + (row * N) + col, c_frag, N, wmma::mem_row_major);
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    half *hostMatA, *hostMatB;
    float* hostMatC;
    half *deviceMatA, *deviceMatB;
    float* deviceMatC;
    initializeMatrix<half>(hostMatA, deviceMatA, M, K, __float2half(2.0f));
    initializeMatrix<half>(hostMatB, deviceMatB, K, N, __float2half(3.0f));
    initializeMatrix<float>(hostMatC, deviceMatC, M, N, __float2half(0.0f));

    const int warpsPerBlock = 4;
    dim3 blockDim(32, warpsPerBlock);

    // Grid maps warps to tiles.
    // X dimension maps to columns (pad_N / 16).
    // Y dimension maps to rows, divided by warps per block.
    dim3 gridDim(N / WMMA_N, (M + (WMMA_M * warpsPerBlock) - 1) / (WMMA_M * warpsPerBlock));

    // Launch Kernel
    wmmaMatmulAnySize<<<gridDim, blockDim>>>(deviceMatA, deviceMatB, deviceMatC, M, N, K);
    cudaDeviceSynchronize();

    // Copy result back from Device to Host, stripping away the padding
    cudaMemcpy2D(hostMatC, N * sizeof(float), deviceMatC, N * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost);

    // Verify output. C should equal K * 2.0 * 3.0 = 50 * 6.0 = 300.0
    std::cout << "Top-left element: " << hostMatC[0] << std::endl;
    std::cout << "Bottom-right element: " << hostMatC[(M - 1) * N + (N - 1)] << std::endl;

    if (hostMatC[0] == 300.0f) {
        std::cout << "Success! Arbitrary size matmul matched expected value." << std::endl;
    }

    freeMatrix<half>(hostMatA, deviceMatA);
    freeMatrix<half>(hostMatB, deviceMatB);
    freeMatrix<float>(hostMatC, deviceMatC);

    return 0;
}
