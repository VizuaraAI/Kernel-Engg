#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda/ptx>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector>
#include "../common/cuda_utils.cuh"

using namespace nvcuda;
namespace ptx = cuda::ptx;

constexpr int WARP_SIZE = 32;

// Block tile dimensions and chunk width (BK=32 -> 64B A-rows, 2x compute per stage)
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;

// Multi-stage ring buffer (4 * (4 KB A + 4 KB B) = 32 KB static SMEM)
constexpr int NUM_STAGES = 4;

// Tensor Core (WMMA) tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warp tile dimensions: each consumer warp covers a 32 x 32 region (2 x 2 WMMA tiles)
// for 2x register reuse of both a_frag and b_frag
constexpr int WM = 32;
constexpr int WN = 32;
constexpr int WARP_TILES_M = WM / WMMA_M;
constexpr int WARP_TILES_N = WN / WMMA_N;

// Warp-specialized block layout:
// - 1 producer warp (32 threads; 1 elected thread issues TMA bulk copies)
// - 4 consumer warps (128 threads) executing WMMA Tensor Core compute
constexpr int NUM_PRODUCER_WARPS = 1;
constexpr int NUM_CONSUMER_WARPS = (BM / WM) * (BN / WN);
constexpr int N_PRODUCERS = NUM_PRODUCER_WARPS * WARP_SIZE;
constexpr int N_CONSUMERS = NUM_CONSUMER_WARPS * WARP_SIZE;
constexpr int NT = N_PRODUCERS + N_CONSUMERS;

// Elect a single thread in warp 0 to issue TMA bulk copies
__device__ __forceinline__ bool isElected() {
    int warp_id = threadIdx.x / WARP_SIZE;
    int uniform_warp_id = __shfl_sync(0xFFFFFFFFu, warp_id, 0);
    return (uniform_warp_id == 0) && elect_one_sync();
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* smem_bar, uint32_t phase) {
    while (!ptx::mbarrier_try_wait_parity(smem_bar, phase)) {
    }
}

// Issue 1D TMA bulk copies for a BM x BK row-major tile of A into tileA[BM][BK].
// Must be called by a single elected thread.
__device__ __forceinline__ void load_tileA_tma(half (*tileA)[BK], const half* matA,
                                               int bk_offset, int block_row, int k,
                                               uint64_t* barrier) {
    constexpr uint32_t row_bytes = BK * sizeof(half); // 64 bytes (16-byte aligned)
#pragma unroll
    for (int row = 0; row < BM; ++row) {
        ptx::cp_async_bulk(ptx::space_shared, ptx::space_global,
                           &tileA[row][0],
                           &matA[(size_t)(block_row + row) * k + bk_offset],
                           row_bytes,
                           barrier);
    }
}

// Issue 1D TMA bulk copies for a BK x BN row-major tile of B into tileB[BK][BN].
// Must be called by a single elected thread.
__device__ __forceinline__ void load_tileB_tma(half (*tileB)[BN], const half* matB,
                                               int bk_offset, int block_col, int n,
                                               uint64_t* barrier) {
    constexpr uint32_t row_bytes = BN * sizeof(half); // 128 bytes (16-byte aligned)
#pragma unroll
    for (int row = 0; row < BK; ++row) {
        ptx::cp_async_bulk(ptx::space_shared, ptx::space_global,
                           &tileB[row][0],
                           &matB[(size_t)(bk_offset + row) * n + block_col],
                           row_bytes,
                           barrier);
    }
}

__device__ void produce(uint64_t ready[], uint64_t filled[],
                        half tileA[NUM_STAGES][BM][BK], half tileB[NUM_STAGES][BK][BN],
                        const half* __restrict__ matA, const half* __restrict__ matB,
                        int block_row, int block_col, int n, int k) {
    if (!isElected()) {
        return;
    }

    constexpr uint32_t tx_bytes = sizeof(tileA[0]) + sizeof(tileB[0]);
    int stage = 0;
    // ready[s] starts at phase 0, so waiting on phase 1 succeeds immediately on the first wrap
    uint32_t ready_phase = 1;

    for (int bk_offset = 0; bk_offset < k; bk_offset += BK) {
        // Wait until all consumer warps have finished reading this stage
        mbarrier_wait_parity(&ready[stage], ready_phase);

        // Arrive + set expected TMA tx bytes on filled[stage] (expected count = 1)
        (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared,
                                             &filled[stage], tx_bytes);
        load_tileA_tma(tileA[stage], matA, bk_offset, block_row, k, &filled[stage]);
        load_tileB_tma(tileB[stage], matB, bk_offset, block_col, n, &filled[stage]);

        if (++stage == NUM_STAGES) {
            stage = 0;
            ready_phase ^= 1u;
        }
    }
}

__device__ void consume(uint64_t ready[], uint64_t filled[],
                        const half tileA[NUM_STAGES][BM][BK],
                        const half tileB[NUM_STAGES][BK][BN],
                        float* __restrict__ matC,
                        int block_row, int block_col, int consumer_warp_id,
                        int n, int k) {
    constexpr int warps_per_row = BN / WN;
    const int warp_grid_row = consumer_warp_id / warps_per_row; // w.r.t. consumer warp grid
    const int warp_grid_col = consumer_warp_id % warps_per_row;
    const int warp_row = warp_grid_row * WM;                    // w.r.t. thread block
    const int warp_col = warp_grid_col * WN;

    const bool warp_elected = elect_one_sync();

    // Accumulator fragments owned by this consumer warp (2 x 2 WMMA tiles)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WARP_TILES_M][WARP_TILES_N];
#pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int stage = 0;
    uint32_t filled_phase = 0;

    for (int bk_offset = 0; bk_offset < k; bk_offset += BK) {
        // Wait until producer TMA transfers for this stage have completed (no consumer arrival)
        mbarrier_wait_parity(&filled[stage], filled_phase);

        // Shared -> WMMA fragments -> Tensor Core MMA
#pragma unroll
        for (int dot_k = 0; dot_k < BK; dot_k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                a_frag[WARP_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                b_frag[WARP_TILES_N];

#pragma unroll
            for (int i = 0; i < WARP_TILES_M; ++i) {
                wmma::load_matrix_sync(a_frag[i], &tileA[stage][warp_row + i * WMMA_M][dot_k], BK);
            }
#pragma unroll
            for (int j = 0; j < WARP_TILES_N; ++j) {
                wmma::load_matrix_sync(b_frag[j], &tileB[stage][dot_k][warp_col + j * WMMA_N], BN);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILES_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILES_N; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        // Release this stage once all 32 lanes in this warp have finished reading SMEM
        __syncwarp();
        if (warp_elected) {
            (void)ptx::mbarrier_arrive(&ready[stage]);
        }

        if (++stage == NUM_STAGES) {
            stage = 0;
            filled_phase ^= 1u;
        }
    }

    // Write accumulator fragments back to global memory
#pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {
            const int abs_row = block_row + warp_row + i * WMMA_M;
            const int abs_col = block_col + warp_col + j * WMMA_N;
            wmma::store_matrix_sync(&matC[(size_t)abs_row * n + abs_col],
                                    c_frag[i][j], n, wmma::mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(NT)
    matmul(const half* __restrict__ matA, const half* __restrict__ matB, float* __restrict__ matC,
           int m, int n, int k) {
    __shared__ alignas(128) half tileA[NUM_STAGES][BM][BK];
    __shared__ alignas(128) half tileB[NUM_STAGES][BK][BN];
    __shared__ alignas(8) uint64_t ready[NUM_STAGES];
    __shared__ alignas(8) uint64_t filled[NUM_STAGES];

    const int tid = threadIdx.x;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < NUM_STAGES; ++s) {
            ptx::mbarrier_init(&ready[s], NUM_CONSUMER_WARPS);
            ptx::mbarrier_init(&filled[s], 1);
        }
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    const int block_row = blockIdx.y * BM; // w.r.t. grid
    const int block_col = blockIdx.x * BN;

    if (tid < N_PRODUCERS) {
        produce(ready, filled, tileA, tileB, matA, matB, block_row, block_col, n, k);
    } else {
        const int consumer_warp_id = (tid - N_PRODUCERS) / WARP_SIZE;
        consume(ready, filled, tileA, tileB, matC, block_row, block_col, consumer_warp_id, n, k);
    }
}

int main() {
    // NOTE: kernel assumes m % BM == 0, n % BN == 0, k % BK == 0 (no bounds checks)
    const int m = 4096, k = 4096, n = 4096;

    std::vector<half> hA((size_t)m * k, __float2half(2.0f));
    std::vector<half> hB((size_t)k * n, __float2half(3.0f));
    std::vector<float> hC((size_t)m * n, 0.0f);

    half *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));

    dim3 threads_per_block(NT, 1);
    dim3 blocks_per_grid(CEIL_DIV(n, BN), CEIL_DIV(m, BM));

    matmul<<<blocks_per_grid, threads_per_block>>>(dA, dB, dC, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: every element should equal 2 * 3 * k
    const float expected = 2.0f * 3.0f * (float)k;
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
