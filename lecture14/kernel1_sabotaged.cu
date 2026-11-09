#include <iostream>
#include <cuda_runtime.h>

// Problem: Matrix Multiplication C = A * B
// Dimensions: 1024 x 1024
#define N 1024

// =============================================================================
// KERNEL 1 (SABOTAGED): Uncoalesced Global Memory Access
//
// NVIDIA Triage Path: Decision 2 -> Decision 5 -> Decision 6
//
// Root Cause:
//   threadIdx.x is mapped to rows instead of columns:
//     row = blockIdx.x * blockDim.x + threadIdx.x;
//   Consecutive threads in a warp access elements separated by stride N (4096 bytes).
//
// NCU Diagnostic Signatures:
//   - Speed of Light: Memory Throughput (L1/TEX) ~95-99%, Compute Throughput < 6%
//   - Memory Workload Analysis: Sectors/request ~16.5-32 (ideal is 4)
//   - Bytes per sector: Only 4.0 out of 32 bytes utilized (12.5%)
//   - Warp State Statistics: Dominated by "Stall LG Throttle" (>85% of warp stalls)
//   - Source Page: LDG.E instruction holds the bulk of LG Throttle stall samples
// =============================================================================
__global__ void matmul_sabotaged(const float* A, const float* B, float* C, int n) {
    // SABOTAGE: threadIdx.x varies across ROWS (stride = 4096 bytes between threads)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

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
    matmul_sabotaged<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    std::cout << "[Kernel 1 SABOTAGED] C[0]: " << h_C[0] << " (Expected: " << (N * 2.0f) << ")\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
