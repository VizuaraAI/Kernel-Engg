/*
Docs:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma
*/
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <iostream>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#define N_ELEMENTS 102400
#define N_ELEMENTS_IN_BLOCK 32

__device__ inline bool isElected() {
    int tid = threadIdx.x;
    int warpId = tid / 32;
    // the variable uniformWarpId for every thread in the warp
    // would contain the quantity held by value from thread 0 in the warp.
    // thread 0 has "broadcast" its value to other threads.
    // https://stackoverflow.com/a/76123988
    int uniformWarpId = __shfl_sync(0xFFFFFFFF, warpId, 0);
    return (uniformWarpId == 0) && (ptx::elect_sync(0xFFFFFFFF));
}

__global__ void addVectorKernel(const float* vec, float* output, const int N) {
    __shared__ alignas(16) float buffer[N_ELEMENTS_IN_BLOCK];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier_t barrier;

    const float* vecBlock = vec + blockIdx.x * N_ELEMENTS_IN_BLOCK;
    float* outputBlock = output + blockIdx.x * N_ELEMENTS_IN_BLOCK;

    if (threadIdx.x == 0) {
        // The barrier is initialized with the number of threads participating in the block.
        // As a result, the barrier will flip only if all threads have arrived on this barrier.
        init(&barrier, N_ELEMENTS_IN_BLOCK);
    }
    __syncthreads();

    if (isElected()) {
        cuda::memcpy_async(buffer, vecBlock, cuda::aligned_size_t<16>(sizeof(buffer)), barrier);

        ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, buffer, vecBlock, sizeof(buffer),
                           cuda::device::barrier_native_handle(barrier));
        cuda::device::barrier_expect_tx(barrier, sizeof(buffer));
    }

    barrier.arrive_and_wait();

    buffer[threadIdx.x] *= 2.0f;

    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();

    if (isElected()) {
        ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, outputBlock, buffer,
                           sizeof(buffer));
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
}

int main() {
    auto* hostVec = new float[N_ELEMENTS];
    auto* hostOutput = new float[N_ELEMENTS];
    for (int i = 0; i < N_ELEMENTS; i++) {
        hostVec[i] = static_cast<float>(i) / 1e+6;
    }

    float* vec;
    float* output;
    cudaMalloc(&vec, N_ELEMENTS * sizeof(float));
    cudaMalloc(&output, N_ELEMENTS * sizeof(float));

    cudaMemcpy(vec, hostVec, N_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N_ELEMENTS_IN_BLOCK, 1, 1);
    dim3 blocksPerGrid((N_ELEMENTS + N_ELEMENTS_IN_BLOCK - 1) / N_ELEMENTS_IN_BLOCK, 1, 1);
    addVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(vec, output, N_ELEMENTS);

    cudaMemcpy(hostOutput, output, N_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_ELEMENTS; i++) {
        assert(fabsf(hostOutput[i] - 2.0f * hostVec[i]) < 1e-6f);
    }

    return 0;
}