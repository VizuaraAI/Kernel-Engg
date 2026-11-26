// kernel 04
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

#include "../common/cuda_utils.cuh"

using namespace cute;

constexpr int WARP_SIZE = 32;

// CTA tile dimensions and multi-stage TMA pipeline depth
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int PIPE_STAGES = 4;

// WGMMA atom: 64x128x16 FP32 += FP16 * FP16, reading A and B directly from shared memory (SS).
//   - A is (M, K) row-major -> K-major (GMMA::Major::K)
//   - B is (K, N) row-major -> viewed as (N, K) with stride (1, N) -> MN-major (GMMA::Major::MN)
// Tiled across 2 warpgroups along M -> 2 * 128 = 256 threads covering a 128 x 128 CTA output tile.
using MmaAtom = SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::MN>;
using TiledMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _1, _1>>{}));

constexpr int NT = decltype(size(TiledMma{}))::value; // 256 threads (2 warpgroups)

// 128B-swizzled shared memory layouts compatible with both 2D TMA and WGMMA descriptors
using SmemLayoutA_Stage = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<cute::half_t>{},
                                                 make_shape(Int<BM>{}, Int<BK>{})));
using SmemLayoutB_Stage = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<cute::half_t>{},
                                                 make_shape(Int<BN>{}, Int<BK>{})));

using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<cute::half_t>{},
                                           make_shape(Int<BM>{}, Int<BK>{}, Int<PIPE_STAGES>{})));
using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<cute::half_t>{},
                                           make_shape(Int<BN>{}, Int<BK>{}, Int<PIPE_STAGES>{})));

static_assert(cosize_v<SmemLayoutA_Stage> == BM * BK);
static_assert(cosize_v<SmemLayoutB_Stage> == BN * BK);
static_assert(cosize_v<SmemLayoutA> == BM * BK * PIPE_STAGES);
static_assert(cosize_v<SmemLayoutB> == BN * BK * PIPE_STAGES);

struct SharedStorage {
    alignas(128) cute::half_t smem_A[cosize_v<SmemLayoutA>];
    alignas(128) cute::half_t smem_B[cosize_v<SmemLayoutB>];
    alignas(8) uint64_t tma_mbar[PIPE_STAGES];
};

// Elect a single thread in warp 0 to issue 2D TMA loads and mbarrier transactions
__device__ __forceinline__ bool isElected() {
    int warp_id = __shfl_sync(0xFFFFFFFFu, static_cast<int>(threadIdx.x / WARP_SIZE), 0);
    return (warp_id == 0) && cute::elect_one_sync();
}

template <class TmaLoadA, class TmaLoadB>
__global__ void __launch_bounds__(NT)
    matmul_wgmma(CUTE_GRID_CONSTANT TmaLoadA const tma_a, CUTE_GRID_CONSTANT TmaLoadB const tma_b,
                 float* __restrict__ matC, int m, int n, int k) {
    extern __shared__ char smem_buf[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    const int tid = threadIdx.x;
    const bool elected = isElected();

    // Initialize mbarrier per pipeline stage and prefetch TMA descriptors
    if (elected) {
        cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
#pragma unroll
        for (int s = 0; s < PIPE_STAGES; ++s) {
            cutlass::arch::ClusterTransactionBarrier::init(&smem.tma_mbar[s], 1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Shared memory tensors: (BM, BK, PIPE_STAGES) and (BN, BK, PIPE_STAGES)
    Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smem_B), SmemLayoutB{});

    // Global TMA coordinate tensors and CTA tiles
    auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);

    Tensor mA = tma_a.get_tma_tensor(make_shape(m, k)); // (M, K)
    Tensor mB = tma_b.get_tma_tensor(make_shape(n, k)); // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr(matC), make_shape(m, n), make_stride(n, _1{}));

    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BM, BK, k_tiles)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BN, BK, k_tiles)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BM, BN)

    // Partition A and B for 2D TMA loads
    auto cta_tma_a = tma_a.get_slice(_0{});
    auto cta_tma_b = tma_b.get_slice(_0{});
    Tensor tAgA = cta_tma_a.partition_S(gA); // (TMA, TMA_M, TMA_K, k_tiles)
    Tensor tAsA = cta_tma_a.partition_D(sA); // (TMA, TMA_M, TMA_K, PIPE_STAGES)
    Tensor tBgB = cta_tma_b.partition_S(gB); // (TMA, TMA_N, TMA_K, k_tiles)
    Tensor tBsB = cta_tma_b.partition_D(sB); // (TMA, TMA_N, TMA_K, PIPE_STAGES)

    // Partition A, B, and C for WGMMA across the 2 warpgroups
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    Tensor tCsA = thr_mma.partition_A(sA);       // (MMA, MMA_M, MMA_K, PIPE_STAGES)
    Tensor tCsB = thr_mma.partition_B(sB);       // (MMA, MMA_N, MMA_K, PIPE_STAGES)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // GMMA descriptors for A
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // GMMA descriptors for B

    Tensor tCgC = thr_mma.partition_C(gC);       // (MMA, MMA_M, MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // FP32 register accumulators (64 floats/thread)
    clear(tCrC);

    constexpr uint32_t kTmaTransactionBytes =
        static_cast<uint32_t>((BM * BK + BN * BK) * sizeof(cute::half_t));

    const int k_tile_count = size<2>(gA);

    auto issue_tma = [&](int k_tile, int stage) {
        if (elected) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.tma_mbar[stage],
                                                                           kTmaTransactionBytes);
            auto tAsA_stage = tAsA(_, _, _, stage);
            auto tBsB_stage = tBsB(_, _, _, stage);
            cute::copy(tma_a.with(smem.tma_mbar[stage]), tAgA(_, _, _, k_tile), tAsA_stage);
            cute::copy(tma_b.with(smem.tma_mbar[stage]), tBgB(_, _, _, k_tile), tBsB_stage);
        }
    };

    // Prefill the first (PIPE_STAGES - 1) stages of the TMA ring buffer
    constexpr int kPrefillStages = PIPE_STAGES - 1;
#pragma unroll
    for (int s = 0; s < kPrefillStages; ++s) {
        if (s < k_tile_count) {
            issue_tma(s, s);
        }
    }

    // Pipelined mainloop: overlap 2D TMA loads with asynchronous WGMMA execution
    int read_stage = 0;
    uint32_t read_phase = 0;

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        // 1. Wait for TMA transfer into read_stage to complete
        cutlass::arch::ClusterTransactionBarrier::wait(&smem.tma_mbar[read_stage], read_phase);

        // 2. Issue asynchronous WGMMA batch directly from swizzled SMEM (4 x k=16 steps)
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        cute::gemm(tiled_mma, tCrA(_, _, _, read_stage), tCrB(_, _, _, read_stage), tCrC);
        warpgroup_commit_batch();
        warpgroup_fence_operand(tCrC);

        // 3. Ensure all warpgroups have finished reading write_stage before reusing it for TMA.
        //    Keeping at most 1 WGMMA batch in flight guarantees the previous stage (write_stage)
        //    has completed while the current read_stage WGMMA stays in flight.
        const int next_k = k_tile + kPrefillStages;
        if (next_k < k_tile_count) {
            warpgroup_wait<1>();
            __syncthreads();
            const int write_stage = next_k % PIPE_STAGES;
            issue_tma(next_k, write_stage);
        }

        // Advance ring buffer cursor and mbarrier parity phase
        if (++read_stage == PIPE_STAGES) {
            read_stage = 0;
            read_phase ^= 1u;
        }
    }

    // Drain all remaining in-flight WGMMA operations before reading accumulators
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // Epilogue: store FP32 register accumulators to global memory
    cute::copy(tCrC, tCgC);
}

int main() {
    // NOTE: kernel assumes m % BM == 0, n % BN == 0, k % BK == 0 (no bounds checks)
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

    // Build 2D TMA descriptors for A (M x K, row-major) and B (K x N, row-major -> N x K col-major)
    Tensor mA = make_tensor(dA, make_shape(m, k), make_stride(k, _1{}));
    Tensor mB = make_tensor(dB, make_shape(n, k), make_stride(_1{}, n));
    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA_Stage{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB_Stage{});

    constexpr int smem_size = static_cast<int>(sizeof(SharedStorage));
    CUDA_CHECK(cudaFuncSetAttribute(matmul_wgmma<decltype(tma_a), decltype(tma_b)>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 threads_per_block(NT, 1);
    dim3 blocks_per_grid(CEIL_DIV(n, BN), CEIL_DIV(m, BM));

    matmul_wgmma<<<blocks_per_grid, threads_per_block, smem_size>>>(tma_a, tma_b, dC, m, n, k);
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
