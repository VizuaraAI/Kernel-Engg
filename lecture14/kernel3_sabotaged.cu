#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// Problem: Heavy elementwise compute pipeline on 16M elements
#define N (16 * 1024 * 1024)

// =============================================================================
// KERNEL 3 (SABOTAGED): Severe Grid Starvation (Workload Too Small for GPU)
//
// NVIDIA Triage Path: Decision 1 -> Action: Scale across blocks first!
//
// Root Cause:
//   Launching only 4 thread blocks for the entire GPU:
//     gridDim = 4, blockDim = 256 (only 1,024 active threads across whole GPU).
//   Threads serialize work in a massive grid-stride loop (16,384 elements per thread).
//   On an H100 (132 SMs), 128 SMs (>97%) remain completely idle!
//
// NCU Diagnostic Signatures:
//   - Launch Statistics: Waves Per Multiprocessor (launch__waves_per_multiprocessor) = 0.030 << 1.0
//   - Workload Distribution: sm__cycles_active.min == 0 (over 95% idle SMs)
//   - Speed of Light: BOTH Compute SOL (~3%) and Memory SOL (~2%) are extremely low
//   - NVIDIA Triage Directive:
//       "Scale independent work across blocks first. Typical refactor: parallelize
//        independent outer-loop work across blocks instead of serializing inside one block."
// =============================================================================
__global__ void compute_pipeline_sabotaged(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // SABOTAGE: Serializing outer loop inside 4 thread blocks
    for (int i = idx; i < n; i += stride) {
        float x = in[i];
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            x = sinf(x) * cosf(x) + sqrtf(fabsf(x) + 1.0f);
        }
        out[i] = x;
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

    // SABOTAGE: Only 4 thread blocks launched!
    dim3 block(256);
    dim3 grid(4);

    // Launch kernel once
    compute_pipeline_sabotaged<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    std::cout << "[Kernel 3 SABOTAGED] out[0]: " << h_out[0] << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
