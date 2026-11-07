#include <iostream>
#include <cuda_runtime.h>

#define N 10000

// Ceiling division: divides `a` by `b` and rounds UP to the next integer.
// Plain C++ integer division (10000 / 1024 = 9) truncates the remainder,
// which would leave the last 784 elements unprocessed. Adding (b - 1) before
// dividing nudges any non-zero remainder up to the next whole block (10).
#define CEILDIV(a, b) ((a + b - 1) / b)

/*
Elementwise vector addition on the GPU:
    a = [a0, a1, a2, ...]   (N = 10,000 floats)
    b = [b0, b1, b2, ...]   (N = 10,000 floats)
    c = a + b = [a0 + b0, a1 + b1, ...]

Unlike CPU code, `vecadd` does not use a `for` loop over the N output elements.
Instead, we launch >= N GPU threads in parallel, and each thread computes ONE
output element (other kernels like MaxPool or MatMul may still loop within a
thread when a single output element combines multiple inputs).
*/

// `__global__` marks a kernel function launched from the CPU and executed on
// the GPU. Parameters `vecA`, `vecB`, and `vecC` are 64-bit pointers holding
// starting memory addresses inside GPU HBM (VRAM) so each thread knows where
// the arrays live in device memory.
__global__ void vecadd(
    const float* vecA, // base address of input array A in GPU HBM
    const float* vecB, // base address of input array B in GPU HBM
    float* vecC        // base address of output array C in GPU HBM
) {
    // `blockDim`, `blockIdx`, and `threadIdx` are read-only built-in hardware
    // variables automatically provided inside every `__global__` kernel from
    // the `<<<blocksPerGrid, threadsPerBlock>>>` launch configuration:
    //   - blockDim.x:  number of threads per block (1,024 here)
    //   - blockIdx.x:  which block this thread belongs to (0 .. 9)
    //   - threadIdx.x: this thread's index inside its block (0 .. 1,023)
    int globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

    // 10 blocks * 1,024 threads = 10,240 launched threads for N = 10,000
    // elements. Guard the last 240 threads in block 9 from accessing memory
    // past the end of the arrays.
    if (globalThreadIdx >= N) {
        return;
    }

    // Although these two lines look like simple array indexing, under the hood
    // they trigger the full HBM <-> SM data transport and compute pipeline:
    //   1. LDG (Load Global):  Transport 4 bytes of `vecA[globalThreadIdx]`
    //      from GPU HBM -> L2 -> L1 -> SM register (e.g., R2).
    //   2. LDG (Load Global):  Transport 4 bytes of `vecB[globalThreadIdx]`
    //      from GPU HBM -> L2 -> L1 -> SM register (e.g., R3).
    //   3. FADD (FP32 Add):    CUDA core adds R2 + R3 -> SM register R4 (`sum`).
    //   4. STG (Store Global): Transport 4 bytes of `sum` (R4) from the SM
    //      register -> L1 -> L2 -> GPU HBM at `vecC[globalThreadIdx]`.
    float sum = vecA[globalThreadIdx] + vecB[globalThreadIdx];
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

    // 1. Host vectors (allocated in CPU DRAM)
    // `hostVecA`, `hostVecB`, `hostVecC` hold CPU RAM addresses.
    float* hostVecA = new float[N];
    float* hostVecB = new float[N];
    float* hostVecC = new float[N];
    populateVector(hostVecA, N, 3.14f);
    populateVector(hostVecB, N, 2.0f);

    // 2. Device vectors (allocated in GPU HBM / VRAM)
    // `deviceVecA`, `deviceVecB`, `deviceVecC` live on the CPU stack, but the
    // 64-bit addresses stored inside them point to locations in GPU HBM.
    float* deviceVecA;
    float* deviceVecB;
    float* deviceVecC;
    cudaMalloc(&deviceVecA, N * sizeof(float));
    cudaMalloc(&deviceVecB, N * sizeof(float));
    cudaMalloc(&deviceVecC, N * sizeof(float));

    // 3. Copy input arrays from CPU DRAM to GPU HBM before launching the kernel
    cudaMemcpy(deviceVecA, hostVecA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVecB, hostVecB, N * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Configure the 3D launch geometry (using 1D x-axis; y=1, z=1)
    // Named `blocksPerGrid` and `threadsPerBlock` on the host so they do not
    // shadow CUDA's built-in `gridDim` and `blockDim` variables in the kernel.
    int numThreadsInBlock = 1024;
    int numThreadBlocksInGrid = CEILDIV(N, numThreadsInBlock);
    dim3 blocksPerGrid(numThreadBlocksInGrid, 1, 1);
    dim3 threadsPerBlock(numThreadsInBlock, 1, 1);

    // 5. Launch the kernel on the GPU:
    //   - `<<<blocksPerGrid, threadsPerBlock>>>` configures the GPU hardware
    //     launcher (which populates `gridDim`, `blockDim`, `blockIdx`, and
    //     `threadIdx` inside `vecadd`).
    //   - `(deviceVecA, deviceVecB, deviceVecC)` passes the three 8-byte GPU
    //     HBM base addresses to the kernel parameters.
    vecadd<<<blocksPerGrid, threadsPerBlock>>>(deviceVecA, deviceVecB, deviceVecC);
    cudaDeviceSynchronize();

    // 6. Copy the output array from GPU HBM back to CPU DRAM and print
    cudaMemcpy(hostVecC, deviceVecC, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        std::cout << hostVecC[i] << ' ';
    }
    std::cout << '\n';

    return 0;
}

}