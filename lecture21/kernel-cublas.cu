#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include "../common/cuda_utils.cuh"

#define CUBLAS_CHECK(call)                                                              \
    do {                                                                                \
        cublasStatus_t status_ = (call);                                                \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                         \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)status_, __FILE__,       \
                    __LINE__);                                                          \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

void matmul_cublas(cublasHandle_t handle, const half* dA, const half* dB, float* dC, int m,
                   int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              n, m, k,                        // dims of C^T (n x m), inner dim k
                              &alpha,
                              dB, CUDA_R_16F, n,              // B^T as column-major: (n x k), ld = n
                              dA, CUDA_R_16F, k,              // A^T as column-major: (k x m), ld = k
                              &beta,
                              dC, CUDA_R_32F, n,              // C^T as column-major: (n x m), ld = n
                              CUBLAS_COMPUTE_32F,             // FP32 accumulation on FP16 Tensor Cores
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main() {
    // Unlike the hand-written kernels, cuBLAS has no divisibility requirements.
    const int m = 4096, k = 4096, n = 4096;

    std::vector<half> hA((size_t)m * k, __float2half(2.0f));
    std::vector<half> hB((size_t)k * n, __float2half(3.0f));
    std::vector<float> hC((size_t)m * n, 0.0f);

    half *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Warm-up run (cuBLAS initialisation / kernel selection overhead)
    matmul_cublas(handle, dA, dB, dC, m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    matmul_cublas(handle, dA, dB, dC, m, n, k);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double tflops = 2.0 * m * n * (double)k / (ms * 1e-3) / 1e12;
    printf("cuBLAS GemmEx (FP16->FP32): %.3f ms, %.1f TFLOP/s\n", ms, tflops);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: every element should equal 2 * 3 * k
    const float expected = 2.0f * 3.0f * (float)k;
    const float rtol = 1e-5f, atol = 1e-5f;
    size_t errors = 0;
    for (size_t i = 0; i < hC.size(); ++i) {
        if (fabsf(hC[i] - expected) > atol + rtol * fabsf(expected)) {
            if (errors < 10)
                fprintf(stderr, "Mismatch at %zu: got %f, expected %f\n", i, hC[i], expected);
            ++errors;
        }
    }
    printf(errors == 0 ? "PASSED\n" : "FAILED (%zu mismatches)\n", errors);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return errors == 0 ? 0 : 1;
}
