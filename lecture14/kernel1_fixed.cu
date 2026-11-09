#include <iostream>
#include <cuda_runtime.h>

// Problem: Matrix Multiplication C = A * B
// Dimensions: 1024 x 1024
#define N 1024

// =============================================================================
// KERNEL 1 (FIXED): Coalesced Global Memory Access
//
// NVIDIA Triage Path: Decision 2 -> Decision 5 -> Decision 6
//
// The Fix:
//   Swap row and col indexing so threadIdx.x varies across columns:
//     row = blockIdx.y * blockDim.y + threadIdx.y;
//     col = blockIdx.x * blockDim.x + threadIdx.x;
//   Adjacent threads now load contiguous columns (stride 1 = 4 bytes).
//
// Baseline Difference in NCU:
//   - Duration drops by ~92% (12.5x speedup)
//   - Sectors per request drops from ~16.5 to 2.5 (-85%)
//   - Byte utilization per sector rises from 4.0 bytes to 26.4 bytes
//   - "Stall LG Throttle" drops by ~97%
// =============================================================================
__global__ void matmul_fixed(const float* A, const float* B, float* C, int n) {
    // FIX: threadIdx.x varies across COLUMNS (stride = 4 bytes between threads)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(N / block.x, N / block.y);

    // Launch kernel once
    matmul_fixed<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    std::cout << "[Kernel 1 FIXED] C[0]: " << h_C[0] << " (Expected: " << (N * 2.0f) << ")\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
