#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Tensor Core tile dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel expects PADDED dimensions (guaranteed multiples of 16)
__global__ void wmma_matmul_any_size(half *a, half *b, float *c, int pad_M, int pad_N, int pad_K) {
    // 1. Map threads to 16x16 output tiles
    // blockDim.x = 32 (1 warp). blockDim.y = warps per block.
    int warpM = blockIdx.y * blockDim.y + threadIdx.y; // Tile Row
    int warpN = blockIdx.x;                            // Tile Col

    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;

    // Exit if warp is mapped completely outside the padded matrix
    if (row >= pad_M || col >= pad_N) return;

    // 2. Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // 3. Loop over the K dimension, sliding the 16x16 window
    for (int k = 0; k < pad_K; k += WMMA_K) {
        // Load tiles. The trailing argument is the leading dimension (stride) of the padded matrix.
        wmma::load_matrix_sync(a_frag, a + (row * pad_K) + k, pad_K);
        wmma::load_matrix_sync(b_frag, b + (k * pad_N) + col, pad_N);
        
        // Accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4. Store the resulting tile back to global memory
    wmma::store_matrix_sync(c + (row * pad_N) + col, c_frag, pad_N, wmma::mem_row_major);
}

int main() {
    int M = 1200, N = 800, K = 1000;

    int pad_M = (M + 15) / 16 * 16;
    int pad_N = (N + 15) / 16 * 16;
    int pad_K = (K + 15) / 16 * 16;

    // host data initialization
    half *h_a = new half[M * K];
    half *h_b = new half[K * N];
    float *h_c = new float[M * N];
    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(2.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(3.0f);

    // device data initialization
    half *d_a, *d_b; float *d_c;
    cudaMalloc(&d_a, pad_M * pad_K * sizeof(half));
    cudaMalloc(&d_b, pad_K * pad_N * sizeof(half));
    cudaMalloc(&d_c, pad_M * pad_N * sizeof(float));
    cudaMemset(d_a, 0, pad_M * pad_K * sizeof(half));
    cudaMemset(d_b, 0, pad_K * pad_N * sizeof(half));

    // Copy Host to Device using 2D copy to insert padding
    cudaMemcpy2D(d_a, pad_K * sizeof(half), h_a, K * sizeof(half), K * sizeof(half), M, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_b, pad_N * sizeof(half), h_b, N * sizeof(half), N * sizeof(half), K, cudaMemcpyHostToDevice);

    // launch the kernel
    int warps_per_block = 4;
    dim3 blockDim(32, warps_per_block);
    
    // Grid maps warps to tiles. 
    // X dimension maps to columns (pad_N / 16). 
    // Y dimension maps to rows, divided by warps per block.
    dim3 gridDim(
        pad_N / WMMA_N, 
        (pad_M + (WMMA_M * warps_per_block) - 1) / (WMMA_M * warps_per_block)
    );

    // Launch Kernel
    wmma_matmul_any_size<<<gridDim, blockDim>>>(d_a, d_b, d_c, pad_M, pad_N, pad_K);
    cudaDeviceSynchronize();

    // Copy result back from Device to Host, stripping away the padding
    cudaMemcpy2D(h_c, N * sizeof(float), d_c, pad_N * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost);

    // Verify output. C should equal K * 2.0 * 3.0 = 50 * 6.0 = 300.0
    std::cout << "Top-left element: " << h_c[0] << std::endl;
    std::cout << "Bottom-right element: " << h_c[(M - 1) * N + (N - 1)] << std::endl;

    if (h_c[0] == 300.0f) {
        std::cout << "Success! Arbitrary size matmul matched expected value." << std::endl;
    }

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;

    return 0;
}