#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Tensor Cores operate on specific tile sizes. 16x16x16 is standard for FP16.
const int M = 16;
const int N = 16;
const int K = 16;

__global__ void wmma_single_warp_kernel(half *a, half *b, float *c) {
    // 1. Declare WMMA fragments (registers dedicated to Tensor Core operations)
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // 2. Initialize the accumulator fragment to 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 3. Load data from global memory into the fragments
    // The last argument is the leading dimension (stride) of the matrix
    wmma::load_matrix_sync(a_frag, a, K);
    wmma::load_matrix_sync(b_frag, b, K);

    // 4. Perform the matrix multiply-accumulate (D = A * B + C)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 5. Store the result from the fragment back to global memory
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

int main() {
    // Matrix dimensions
    int a_size = M * K;
    int b_size = K * N;
    int c_size = M * N;

    // Host memory
    half *h_a = new half[a_size];
    half *h_b = new half[b_size];
    float *h_c = new float[c_size];

    // Initialize matrices (A = 2.0, B = 3.0)
    for (int i = 0; i < a_size; i++) h_a[i] = __float2half(2.0f);
    for (int i = 0; i < b_size; i++) h_b[i] = __float2half(3.0f);
    for (int i = 0; i < c_size; i++) h_c[i] = 0.0f;

    // Device memory
    half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, a_size * sizeof(half));
    cudaMalloc(&d_b, b_size * sizeof(half));
    cudaMalloc(&d_c, c_size * sizeof(float));

    // Copy to device
    cudaMemcpy(d_a, h_a, a_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_size * sizeof(half), cudaMemcpyHostToDevice);

    // Launch configuration: 1 block, 32 threads (exactly 1 warp)
    dim3 gridDim(1);
    dim3 blockDim(32);

    wmma_single_warp_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, c_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify output (16 * 2.0 * 3.0 = 96.0)
    std::cout << "Top-left element result: " << h_c[0] << std::endl;
    if (h_c[0] == 96.0f) {
        std::cout << "Success! Tensor Core matmul executed correctly." << std::endl;
    }

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;

    return 0;
}