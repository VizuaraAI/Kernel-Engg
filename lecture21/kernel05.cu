// kernel 05
#include <cmath>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/half.h>

#include "../common/cuda_utils.cuh"

using namespace cute;
namespace cg = cooperative_groups;

constexpr int WARP_SIZE = 32;
constexpr int WARPGROUP_SIZE = 128;

// CTA tile dimensions and 3-stage pipeline (48 KB A + 96 KB B = 144 KB dynamic SMEM)
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int NUM_STAGES = 3;

// Thread block cluster: 2 x 1 (2 CTAs share the 256 x 64 tile of B via 2D TMA multicast)
constexpr int CLUSTER_M = 2;
constexpr int CLUSTER_N = 1;
constexpr int CLUSTER_SIZE = CLUSTER_M * CLUSTER_N;

// WGMMA atom: 64x256x16 FP32 += FP16 * FP16, reading A (K-major) and B (MN-major) from SMEM.
// Tiled across 2 consumer warpgroups along M -> 128 x 256 CTA output tile.
using MmaAtom = SM90_64x256x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::MN>;
using TiledMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _1, _1>>{}));

constexpr int NUM_PRODUCER_WARPGROUPS = 1;
constexpr int NUM_CONSUMER_WARPGROUPS = decltype(size(TiledMma{}))::value / WARPGROUP_SIZE; // 2
constexpr int NT = (NUM_PRODUCER_WARPGROUPS + NUM_CONSUMER_WARPGROUPS) * WARPGROUP_SIZE;    // 384

// 128B-swizzled shared memory layouts (single-stage for TMA descriptors, multi-stage for SMEM)
using SmemLayoutAStage = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<cute::half_t>{},
    make_shape(Int<BM>{}, Int<BK>{})));
using SmemLayoutBStage = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<cute::half_t>{},
    make_shape(Int<BN>{}, Int<BK>{})));

using SmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<cute::half_t>{},
    make_shape(Int<BM>{}, Int<BK>{}, Int<NUM_STAGES>{})));
using SmemLayoutB = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<cute::half_t>{},
    make_shape(Int<BN>{}, Int<BK>{}, Int<NUM_STAGES>{})));

static_assert(cosize_v<SmemLayoutA> == BM * BK * NUM_STAGES);
static_assert(cosize_v<SmemLayoutB> == BN * BK * NUM_STAGES);

struct SharedStorage {
    alignas(128) cute::half_t smem_A[cosize_v<SmemLayoutA>];
    alignas(128) cute::half_t smem_B[cosize_v<SmemLayoutB>];
    alignas(8) uint64_t full_mbar[NUM_STAGES];
    alignas(8) uint64_t empty_mbar[NUM_STAGES];
};

// Elect a single thread in warp 0 of the producer warpgroup
__device__ __forceinline__ bool isProducerElected() {
    int warp_id = __shfl_sync(0xFFFFFFFFu, static_cast<int>(threadIdx.x / WARP_SIZE), 0);
    return (warp_id == 0) && cute::elect_one_sync();
}

// L2-friendly swizzled persistent cluster-tile decoder
__device__ __forceinline__ void decode_cluster_tile(int tile_idx, int m_clusters, int n_clusters,
                                                    uint32_t rank_m, int& block_m, int& block_n) {
    constexpr int GROUP_M = 4;
    const int tiles_per_group = GROUP_M * n_clusters;
    const int group_id = tile_idx / tiles_per_group;
    const int first_cm = group_id * GROUP_M;
    const int group_size_m = min(GROUP_M, m_clusters - first_cm);
    const int in_group = tile_idx - group_id * tiles_per_group;
    const int cm = first_cm + (in_group % group_size_m);
    const int cn = in_group / group_size_m;
    block_m = cm * CLUSTER_M + static_cast<int>(rank_m);
    block_n = cn;
}

template <class TmaLoadA, class TmaLoadB>
__global__ void __launch_bounds__(NT, 1) __cluster_dims__(CLUSTER_M, CLUSTER_N, 1)
    matmul(CUTE_GRID_CONSTANT TmaLoadA const tma_a,
           CUTE_GRID_CONSTANT TmaLoadB const tma_b,
           float* __restrict__ matC,
           int m, int n, int k) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    auto cluster = cg::this_cluster();
    const uint32_t rank_m = cluster.block_rank();
    const int cluster_id = static_cast<int>(blockIdx.x / CLUSTER_SIZE);
    const int num_clusters = static_cast<int>(gridDim.x / CLUSTER_SIZE);

    const int tid = threadIdx.x;
    const int wg_idx = tid / WARPGROUP_SIZE;

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
#pragma unroll
        for (int s = 0; s < NUM_STAGES; ++s) {
            cutlass::arch::ClusterTransactionBarrier::init(&smem.full_mbar[s], 1);
            cutlass::arch::ClusterBarrier::init(&smem.empty_mbar[s],
                                                NUM_CONSUMER_WARPGROUPS * CLUSTER_SIZE);
        }
        cutlass::arch::fence_barrier_init();
    }
    cluster.sync();

    // Multi-stage shared memory tensors: (BM, BK, NUM_STAGES) and (BN, BK, NUM_STAGES)
    Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smem_B), SmemLayoutB{});

    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
    Tensor mA = tma_a.get_tma_tensor(make_shape(m, k)); // (M, K)
    Tensor mB = tma_b.get_tma_tensor(make_shape(n, k)); // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr(matC), make_shape(m, n), make_stride(n, _1{}));

    const int m_clusters = m / (BM * CLUSTER_M);
    const int n_clusters = n / (BN * CLUSTER_N);
    const int total_cluster_tiles = m_clusters * n_clusters;
    const int k_tile_count = k / BK;

    constexpr uint32_t kTmaTransactionBytes =
        static_cast<uint32_t>((BM * BK + BN * BK) * sizeof(cute::half_t));
    constexpr uint16_t kMcastMaskB = static_cast<uint16_t>((1u << CLUSTER_M) - 1u);

    if (wg_idx == 0) {
        // Producer warpgroup: shed registers to consumers
        cutlass::arch::warpgroup_reg_dealloc<40>();

        if (isProducerElected()) {
            auto cta_tma_a = tma_a.get_slice(_0{});
            auto cta_tma_b = tma_b.get_slice(rank_m);
            Tensor tAsA = cta_tma_a.partition_D(sA); // (TMA, TMA_M, TMA_K, NUM_STAGES)
            Tensor tBsB = cta_tma_b.partition_D(sB); // (TMA, TMA_N, TMA_K, NUM_STAGES)

            int stage = 0;
            // empty_mbar[s] starts at phase 0, so waiting on phase 1 passes immediately on wrap 0
            uint32_t empty_phase = 1;

            for (int tile_idx = cluster_id; tile_idx < total_cluster_tiles;
                 tile_idx += num_clusters) {
                int block_m, block_n;
                decode_cluster_tile(tile_idx, m_clusters, n_clusters, rank_m, block_m, block_n);

                auto cta_coord = make_coord(block_m, block_n, _);
                Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
                Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

                Tensor tAgA = cta_tma_a.partition_S(gA); // (TMA, TMA_M, TMA_K, k_tiles)
                Tensor tBgB = cta_tma_b.partition_S(gB); // (TMA, TMA_N, TMA_K, k_tiles)

                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    cutlass::arch::ClusterBarrier::wait(&smem.empty_mbar[stage], empty_phase);

                    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                        &smem.full_mbar[stage], kTmaTransactionBytes);
                    // Unicast A (each CTA loads its own 128 x 64 tile)
                    cute::copy(tma_a.with(smem.full_mbar[stage]),
                               tAgA(_, _, _, k_tile), tAsA(_, _, _, stage));
                    // Split-multicast B (each CTA loads half of 256 x 64 and multicasts to cluster)
                    cute::copy(tma_b.with(smem.full_mbar[stage], kMcastMaskB),
                               tBgB(_, _, _, k_tile), tBsB(_, _, _, stage));

                    if (++stage == NUM_STAGES) {
                        stage = 0;
                        empty_phase ^= 1u;
                    }
                }
            }
        }
    } else {
        // Consumer warpgroups (wg_idx 1 and 2): claim extra registers for 64x256 FP32 accumulators
        cutlass::arch::warpgroup_reg_alloc<232>();

        const int consumer_tid = tid - WARPGROUP_SIZE;
        const uint32_t wg_tid = static_cast<uint32_t>(consumer_tid % WARPGROUP_SIZE);

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(consumer_tid);

        Tensor tCsA = thr_mma.partition_A(sA);       // (MMA, MMA_M, MMA_K, NUM_STAGES)
        Tensor tCsB = thr_mma.partition_B(sB);       // (MMA, MMA_N, MMA_K, NUM_STAGES)
        Tensor tCrA = thr_mma.make_fragment_A(tCsA); // GMMA descriptors for A
        Tensor tCrB = thr_mma.make_fragment_B(tCsB); // GMMA descriptors for B

        int stage = 0;
        uint32_t full_phase = 0;

        for (int tile_idx = cluster_id; tile_idx < total_cluster_tiles; tile_idx += num_clusters) {
            int block_m, block_n;
            decode_cluster_tile(tile_idx, m_clusters, n_clusters, rank_m, block_m, block_n);

            auto cta_coord = make_coord(block_m, block_n, _);
            Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BM, BN)
            Tensor tCgC = thr_mma.partition_C(gC);
            Tensor tCrC = thr_mma.make_fragment_C(tCgC); // FP32 accumulators (128 floats/thread)
            clear(tCrC);

            for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                cutlass::arch::ClusterTransactionBarrier::wait(&smem.full_mbar[stage], full_phase);

                warpgroup_fence_operand(tCrC);
                warpgroup_arrive();
                cute::gemm(tiled_mma, tCrA(_, _, _, stage), tCrB(_, _, _, stage), tCrC);
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                warpgroup_fence_operand(tCrC);

                // Signal every CTA in the cluster that this warpgroup finished reading this stage
                cutlass::arch::ClusterBarrier::arrive(&smem.empty_mbar[stage], wg_tid,
                                                      wg_tid < CLUSTER_SIZE);

                if (++stage == NUM_STAGES) {
                    stage = 0;
                    full_phase ^= 1u;
                }
            }

            // Write FP32 accumulator registers back to global memory
            cute::copy(tCrC, tCgC);
        }
    }

    // Ensure no CTA exits while its cluster peer still has in-flight DSMEM / multicast accesses
    cluster.sync();
}

int main() {
    // NOTE: kernel assumes m % (BM * CLUSTER_M) == 0, n % (BN * CLUSTER_N) == 0, k % BK == 0
    const int m = 4096, k = 4096, n = 4096;

    std::vector<cute::half_t> hA(static_cast<size_t>(m) * k, cute::half_t(2.0f));
    std::vector<cute::half_t> hB(static_cast<size_t>(k) * n, cute::half_t(3.0f));
    std::vector<float> hC(static_cast<size_t>(m) * n, 0.0f);

    cute::half_t *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(cute::half_t)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(cute::half_t)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(cute::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(cute::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));

    // Build 2D TMA descriptors: unicast A (M x K row-major), multicast B (N x K col-major)
    Tensor mA = make_tensor(dA, make_shape(m, k), make_stride(k, _1{}));
    Tensor mB = make_tensor(dB, make_shape(n, k), make_stride(_1{}, n));
    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutAStage{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, mB, SmemLayoutBStage{}, Int<CLUSTER_M>{});

    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    const int num_persistent_ctas = (num_sms / CLUSTER_SIZE) * CLUSTER_SIZE;

    auto* kernel = matmul<decltype(tma_a), decltype(tma_b)>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem_size)));

    kernel<<<num_persistent_ctas, NT, smem_size>>>(tma_a, tma_b, dC, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: every element should equal 2 * 3 * k
    const float expected = 2.0f * 3.0f * static_cast<float>(k);
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

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return errors == 0 ? 0 : 1;
}
