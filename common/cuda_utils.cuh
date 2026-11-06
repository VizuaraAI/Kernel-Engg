#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

#define CEIL_DIV(a, b) ((a + b - 1) / b)

inline void gpuAssert(const cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "GPU Error: " << cudaGetErrorString(code)
                  << " in " << file << ":" << line << std::endl;
        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

template <typename T> void initializeMatrix(T*& hostMatrix, T*& deviceMatrix, const unsigned int dim0, const unsigned int dim1, const T value) {
    hostMatrix = static_cast<T*>(malloc(dim0 * dim1 * sizeof(T)));
    cudaMalloc(&deviceMatrix, dim0 * dim1 * sizeof(T));
    for (unsigned int i = 0; i < dim0; i++) {
        for (unsigned int j = 0; j < dim1; j++) {
            hostMatrix[i * dim1 + j] = value;
        }
    }
    CUDA_CHECK(cudaMemcpy(deviceMatrix, hostMatrix, dim0 * dim1 * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> void freeMatrix(const T* hostMatrix, T* deviceMatrix) {
    delete hostMatrix;
    cudaFree(deviceMatrix);
}

#endif // CUDA_UTILS_CUH