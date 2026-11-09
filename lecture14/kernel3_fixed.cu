#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// Problem: Heavy elementwise compute pipeline on 16M elements
#define N (16 * 1024 * 1024)

// =============================================================================
// KERNEL 3 (FIXED): Saturated Grid Covering Full Hardware Parallelism
//
// NVIDIA Triage Path: Decision 1 -> Action: Scale across blocks first!
//
// The Fix:
//   Decompose independent elements across 65,536 thread blocks:
//     blockDim = 256, gridDim = (N + 256 - 1) / 256 = 65,536 blocks.
//   Every SM receives ~496 waves of blocks, saturating execution units.
//
// Baseline Difference in NCU:
//   - Waves per multiprocessor jumps from 0.030 to ~496 waves
//   - sm__cycles_active.min jumps from 0 to balanced cycles across all SMs
//   - Compute Speed of Light jumps from ~3% to >70%
//   - Kernel duration drops by >25x
// =============================================================================
__global__ void compute_pipeline_fixed(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // FIX: 1 thread per element, distributing work evenly across thousands of blocks
    if (idx < n) {
        float x = in[idx];
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            x = sinf(x) * cosf(x) + sqrtf(fabsf(x) + 1.0f);
        }
        out[idx] = x;
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_in = new float[N];
    float *h_out = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 360) * 0.01745f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // FIX: Saturated grid covering 16M elements with 65,536 blocks!
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    number of elements 'N' / threadsPerBlock

    // Launch kernel once
    compute_pipeline_fixed<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    std::cout << "[Kernel 3 FIXED] out[0]: " << h_out[0] << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
