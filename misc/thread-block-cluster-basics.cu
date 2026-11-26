/**
An introductory example of Thread Block Clusters and Distributed Shared Memory (DSMEM)
in CUDA C++ (requires Hopper / compute capability 9.0+).

A Thread Block Cluster is a group of thread blocks guaranteed to be co-scheduled on the
same GPU Processing Cluster (GPC). Because blocks in a cluster reside on the same GPC,
they can:
  1. Synchronize across blocks using `cluster.sync()` (hardware-accelerated cluster barrier).
  2. Directly access each other's shared memory (Distributed Shared Memory / DSMEM)
     via `cluster.map_shared_rank(smemPtr, targetBlockRank)`.

Cluster dimensions can be specified in two ways:
  - Compile-time: using the `__cluster_dims__(X, Y, Z)` attribute on the `__global__` kernel
    (used in this example).
  - Runtime: using `cudaLaunchKernelEx` with `cudaLaunchAttributeClusterDimension`.

Note: The grid dimensions must be a multiple of the cluster dimensions.
*/
#include <cassert>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>

namespace cg = cooperative_groups;

constexpr int CLUSTER_SIZE = 2;
constexpr int NUM_CLUSTERS = 2;
constexpr int BLOCK_SIZE = 32;
constexpr int NUM_BLOCKS = CLUSTER_SIZE * NUM_CLUSTERS;
constexpr int N_ELEMENTS = NUM_BLOCKS * BLOCK_SIZE;

// Launch with a compile-time cluster size of (CLUSTER_SIZE, 1, 1).
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) clusterExampleKernel(int* output) {
    __shared__ int smem[BLOCK_SIZE];

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int blockRank = cluster.block_rank();
    unsigned int numBlocksInCluster = cluster.num_blocks();

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // 1. Each thread writes a unique value to its own block's shared memory.
    smem[tid] = static_cast<int>(blockIdx.x) * 100 + tid;

    // 2. Synchronize all blocks in the cluster so every block's shared memory is ready
    //    before any peer block reads from it.
    cluster.sync();

    // 3. Map the next block's shared memory in the cluster (ring neighbor) via DSMEM.
    unsigned int peerRank = (blockRank + 1) % numBlocksInCluster;
    int* peerSmem = cluster.map_shared_rank(smem, peerRank);

    // 4. Read directly from the peer block's shared memory and combine with local SMEM.
    int localVal = smem[tid];
    int peerVal = peerSmem[tid];
    output[globalIdx] = localVal + peerVal;

    if (tid == 0) {
        printf("Block %d (cluster rank %u/%u): local=%d, peer(rank %u)=%d, sum=%d\n", blockIdx.x,
               blockRank, numBlocksInCluster, localVal, peerRank, peerVal, output[globalIdx]);
    }

    // 5. Synchronize before exiting so no block terminates and invalidates its shared memory
    //    while another block in the cluster is still reading from it.
    cluster.sync();
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 9) {
        std::cerr << "Thread Block Clusters require compute capability 9.0+ (found " << prop.major
                  << "." << prop.minor << ").\n";
        return 1;
    }

    int* deviceOutput = nullptr;
    cudaMalloc(&deviceOutput, N_ELEMENTS * sizeof(int));

    dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
    dim3 blocksPerGrid(NUM_BLOCKS, 1, 1);
    clusterExampleKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceOutput);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        cudaFree(deviceOutput);
        return 1;
    }

    int hostOutput[N_ELEMENTS];
    cudaMemcpy(hostOutput, deviceOutput, N_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify each element equals localVal + peerVal from the ring neighbor in the same cluster.
    for (int b = 0; b < NUM_BLOCKS; b++) {
        int clusterStart = (b / CLUSTER_SIZE) * CLUSTER_SIZE;
        int peerBlock = clusterStart + ((b - clusterStart + 1) % CLUSTER_SIZE);
        for (int t = 0; t < BLOCK_SIZE; t++) {
            int expected = (b * 100 + t) + (peerBlock * 100 + t);
            assert(hostOutput[b * BLOCK_SIZE + t] == expected);
        }
    }

    std::cout << "All " << N_ELEMENTS << " elements verified successfully.\n";

    cudaFree(deviceOutput);
    return 0;
}
