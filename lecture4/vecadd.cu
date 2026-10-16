#include <iostream>
#include <cuda_runtime.h>

#define N 10000
#define CEILDIV(a, b) ((a + b - 1) / b)

/*

a = [1 2 3 ...... ]   N = 10000
b = [2 3 5 .......]   N = 10000

elementwise addition:

c = a + b = [a1 + b1 a2 + b2 .......]
number of elements in c = N = 10000
*/

__global__ void vecadd(
    const float* vecA, // data in GPU memory for vecA elements
    const float* vecB, // data in GPU memory for vecB elements
    float* vecC
) {
    int globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalThreadIdx >= N) {
        return;
    }
    int sum = vecA[globalThreadIdx] + vecB[globalThreadIdx];
    vecC[globalThreadIdx] = sum;
}

void populateVector(
    float* vec, int n, float k
) {
    for (int i = 0; i < n; i++) {
        vec[i] = k;
    }
}

int main(void) {

    // host vectors (containers)
    /// memory of these containers is in CPU DRAM
    float* hostVecA = new float[N];
    float* hostVecB = new float[N];
    float* hostVecC = new float[N];
    populateVector(hostVecA, N, 3.14f);
    populateVector(hostVecB, N, 2.0f);

    // device vectors (containers)
    // memory of these containers is in GPU DRAM (VRAM)
    float* deviceVecA;
    float* deviceVecB;
    float* deviceVecC;
    cudaMalloc(&deviceVecA, N * sizeof(float));
    cudaMalloc(&deviceVecB, N * sizeof(float));
    cudaMalloc(&deviceVecC, N * sizeof(float));

    // copy data from host buffers to device buffers
    // before running the kernel
    cudaMemcpy(deviceVecA, hostVecA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVecB, hostVecB, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int numThreadsInBlock = 1024;
    int numThreadBlocksInGrid = CEILDIV(N, numThreadsInBlock);
    dim3 gridDim(numThreadBlocksInGrid, 1, 1);
    dim3 blockDim(numThreadsInBlock, 1, 1);
    
    vecadd<<<gridDim, blockDim>>>(deviceVecA, deviceVecB, deviceVecC);
    cudaDeviceSynchronize();

    // printing the results
    cudaMemcpy(hostVecC, deviceVecC, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        std::cout << hostVecC[i] << ' ';
    }
    std::cout << '\n';

    return 0;
}