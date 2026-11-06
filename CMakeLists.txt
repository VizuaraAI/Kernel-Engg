cmake_minimum_required(VERSION 3.22)
project(kernel-engineering-workshop LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# ============ Lecture 4 - First Kernels ===========================
add_executable(
        lecture04-vecadd
        common/cuda_utils.cuh
        lecture4/vecadd.cu
)

# ============ Lecture 12 - Tensor Cores ===========================
add_executable(
        lecture12-tc-01
        common/cuda_utils.cuh
        lecture12/tc-01.cu
)
add_executable(
        lecture12-tc-02
        common/cuda_utils.cuh
        lecture12/tc-02.cu
)