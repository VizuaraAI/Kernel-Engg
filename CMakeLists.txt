cmake_minimum_required(VERSION 3.22)
project(kernel-engineering-workshop LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

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

# ============ Lecture 18 - Flash Attention ========================
add_executable(
        lecture18-fa3
        common/cuda_utils.cuh
        lecture18/flashattention3.cu
)

# ============ Lecture 20 - Flash Attention 4 ========================
add_executable(
        lecture20-fa4
        common/cuda_utils.cuh
        lecture20/flashattention4.cu
)

# ============ Lecture 21 - Beating cuBLAS on H100 ===================
add_executable(
        lecture21-kernel01
        common/cuda_utils.cuh
        lecture21/kernel01.cu
)
add_executable(
        lecture21-kernel02
        common/cuda_utils.cuh
        lecture21/kernel02.cu
)
add_executable(
        lecture21-kernel03
        common/cuda_utils.cuh
        lecture21/kernel03.cu
)

add_executable(
        lecture21-kernel04
        common/cuda_utils.cuh
        lecture21/kernel04.cu
)
target_include_directories(lecture21-kernel04 PRIVATE ${CMAKE_SOURCE_DIR}/external)
set_target_properties(lecture21-kernel04 PROPERTIES CUDA_ARCHITECTURES "90a")

add_executable(
        lecture21-kernel05
        common/cuda_utils.cuh
        lecture21/kernel05.cu
)
target_include_directories(lecture21-kernel05 PRIVATE ${CMAKE_SOURCE_DIR}/external)
set_target_properties(lecture21-kernel05 PROPERTIES CUDA_ARCHITECTURES "90a")

add_executable(
        lecture21-kernel-cublas
        common/cuda_utils.cuh
        lecture21/kernel-cublas.cu
)
target_link_libraries(
        lecture21-kernel-cublas
        PRIVATE
        cublas
)

# ============ misc ================================================
add_executable(
        misc-mbarriers-basics
        misc/async-barrier-basics.cu
)
add_executable(
        misc-mbarriers-producer-consumer
        misc/mbarrier-producer-consumer.cu
)
add_executable(
        misc-tma-basics
        misc/tma-basics.cu
)
add_executable(
        misc-thread-block-cluster-basics
        misc/thread-block-cluster-basics.cu
)