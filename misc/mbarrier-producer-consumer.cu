/*
Docs:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#producer-consumer-pattern-using-barriers
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#producer-consumer-pattern-through-warp-specialization
*/
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <iostream>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

#define N_ELEMENTS 10240
#define N_PRODUCERS 512
#define N_CONSUMERS 512
#define THREADS_PER_BLOCK (N_PRODUCERS + N_CONSUMERS)
#define N_ELEMENTS_PER_BLOCK 1024
#define BUFFER_LEN N_PRODUCERS

__device__ void produce(barrier_t ready[], barrier_t filled[], float smemBuffer[2][BUFFER_LEN],
                        const int bufferLength, const float* gmemData, const int N) {

    for (int i = 0; i < N / bufferLength; i++) {
        ready[i % 2].arrive_and_wait();
        smemBuffer[i % 2][threadIdx.x] = gmemData[i * bufferLength + threadIdx.x];
        filled[i % 2].arrive();
    }
}

__device__ void consume(barrier_t ready[], barrier_t filled[],
                        const float smemBuffer[2][BUFFER_LEN], const int bufferLength,
                        float* gmemResult, const int N) {
    ready[0].arrive();
    ready[1].arrive();
    float sum = 0.0f;
    for (int i = 0; i < N / bufferLength; i++) {
        filled[i % 2].arrive_and_wait();
        sum += smemBuffer[i % 2][threadIdx.x - 512];
        ready[i % 2].arrive();
    }
    atomicAdd(gmemResult, sum);
}

__global__ void barriersProducerConsumerExample(const float data[], float* result) {
    __shared__ barrier_t barriers[4];
    __shared__ float buffer[2][BUFFER_LEN];

    if (threadIdx.x < 4) {
        // the first four threads in the current block will
        // initialize the four barriers.
        // the expected arrival count is blockDim.x
        init(barriers + threadIdx.x, blockDim.x);
    }
    __syncthreads();

    if (threadIdx.x < N_PRODUCERS) {
        const float* blockData = data + blockIdx.x * blockDim.x;
        produce(barriers, barriers + 2, buffer, BUFFER_LEN, blockData, N_ELEMENTS_PER_BLOCK);
    } else {
        consume(barriers, barriers + 2, buffer, BUFFER_LEN, result, N_ELEMENTS_PER_BLOCK);
    }
}

float computeCpuSum(const float* data, const int N) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    auto* hostData = new float[N_ELEMENTS];
    for (int i = 0; i < N_ELEMENTS; i++) {
        hostData[i] = static_cast<float>(i) / 1e+6;
    }
    float* deviceData;
    cudaMalloc(&deviceData, N_ELEMENTS * sizeof(float));
    cudaMemcpy(deviceData, hostData, N_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    float* deviceResult;
    cudaMalloc(&deviceResult, sizeof(float));

    dim3 threadsPerBlock(N_ELEMENTS_PER_BLOCK, 1, 1);
    dim3 blocksPerGrid((N_ELEMENTS + N_ELEMENTS_PER_BLOCK - 1) / N_ELEMENTS_PER_BLOCK, 1, 1);
    barriersProducerConsumerExample<<<blocksPerGrid, threadsPerBlock>>>(deviceData, deviceResult);
    cudaDeviceSynchronize();

    float hostResult;
    cudaMemcpy(&hostResult, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);

    const float cpuSum = computeCpuSum(hostData, N_ELEMENTS);

    std::cout << "[expected] sum = " << cpuSum << "\n";
    std::cout << "[actual] sum = " << hostResult << "\n";

    assert(std::fabs(cpuSum - hostResult) < 1e-2);

    cudaFree(deviceData);
    delete[] hostData;
    return 0;
}