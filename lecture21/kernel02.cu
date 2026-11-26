#include <cmath>
#include <cstdio>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector>
#include "../common/cuda_utils.cuh"

using namespace nvcuda;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int WARP_SIZE = 32;

// Block tile dimensions and chunk width
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

// Tensor Core (WMMA) tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warp tile dimensions: each warp covers a 16 x 32 region (1 x 2 WMMA tiles)
constexpr int WM = 16;
constexpr int WN = 32;
constexpr int WARP_TILES_M = WM / WMMA_M;
constexpr int WARP_TILES_N = WN / WMMA_N;

// Threads per block: (64/16) * (64/32) = 8 warps = 256 threads
constexpr int NUM_WARPS = (BM / WM) * (BN / WN);
constexpr int NT = NUM_WARPS * WARP_SIZE;

__device__ __forceinline__ bool elect_one_sync() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, %1;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred)
        : "r"(0xFFFFFFFFu));
    return pred != 0;
}
// Elect a single thread in warp 0 to issue TMA bulk copies
__device__ __forceinline__ bool isElected() {
    int warp_id = threadIdx.x / WARP_SIZE;
    int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);
    return (uniform_warp_id == 0) && elect_one_sync();
}

// Issue 1D TMA bulk copies for a BM x BK row-major tile of A into tileA[BM][BK].
// Must be called by a single elected thread.
__device__ __forceinline__ void load_tileA_tma(half (*tileA)[BK], const half* matA,
                                               int bk_offset, int block_row, int k,
                                               barrier_t& barrier) {
    constexpr uint32_t row_bytes = BK * sizeof(half); // 32 bytes (16-byte aligned)
#pragma unroll
    for (int row = 0; row < BM; ++row) {
        ptx::cp_async_bulk(ptx::space_shared, ptx::space_global,
                           &tileA[row][0],
                           &matA[(size_t)(block_row + row) * k + bk_offset],
                           row_bytes,
                           cuda::device::barrier_native_handle(barrier));
    }
}

// Issue 1D TMA bulk copies for a BK x BN row-major tile of B into tileB[BK][BN].
// Must be called by a single elected thread.
__device__ __forceinline__ void load_tileB_tma(half (*tileB)[BN], const half* matB,
                                               int bk_offset, int block_col, int n,
                                               barrier_t& barrier) {
    constexpr uint32_t row_bytes = BN * sizeof(half); // 128 bytes (16-byte aligned)
#pragma unroll
    for (int row = 0; row < BK; ++row) {
        ptx::cp_async_bulk(ptx::space_shared, ptx::space_global,
                           &tileB[row][0],
                           &matB[(size_t)(bk_offset + row) * n + block_col],
                           row_bytes,
                           cuda::device::barrier_native_handle(barrier));
    }
}

__global__ void __launch_bounds__(NT)
    matmul(const half* __restrict__ matA, const half* __restrict__ matB, float* __restrict__ matC,
           int m, int n, int k) {
    __shared__ alignas(128) half tileA[BM][BK];
    __shared__ alignas(128) half tileB[BK][BN];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier_t barrier;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;

    if (tid == 0) {
        init(&barrier, NT);
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    constexpr int warps_per_row = BN / WN;
    const int warp_grid_row = warp_id / warps_per_row; // w.r.t. warp grid
    const int warp_grid_col = warp_id % warps_per_row;
    const int warp_row = warp_grid_row * WM;           // w.r.t. thread block
    const int warp_col = warp_grid_col * WN;

    const int block_row = blockIdx.y * BM;             // w.r.t. grid
    const int block_col = blockIdx.x * BN;

    // Accumulator fragments owned by this warp (WARP_TILES_M x WARP_TILES_N)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WARP_TILES_M][WARP_TILES_N];
#pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int bk_offset = 0; bk_offset < k; bk_offset += BK) {
        // Phase 1: global -> shared via TMA
        if (isElected()) {
            cuda::device::barrier_expect_tx(barrier, sizeof(tileA) + sizeof(tileB));
            load_tileA_tma(tileA, matA, bk_offset, block_row, k, barrier);
            load_tileB_tma(tileB, matB, bk_offset, block_col, n, barrier);
        }

        // Wait for all threads and TMA transfers to complete
        barrier.arrive_and_wait();

        // Phase 2: shared -> WMMA fragments -> Tensor Core MMA
#pragma unroll
        for (int dot_k = 0; dot_k < BK; dot_k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                a_frag[WARP_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                b_frag[WARP_TILES_N];

#pragma unroll
            for (int i = 0; i < WARP_TILES_M; ++i) {
                wmma::load_matrix_sync(a_frag[i], &tileA[warp_row + i * WMMA_M][dot_k], BK);
            }
#pragma unroll
            for (int j = 0; j < WARP_TILES_N; ++j) {
                wmma::load_matrix_sync(b_frag[j], &tileB[dot_k][warp_col + j * WMMA_N], BN);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILES_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILES_N; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        // Ensure all warps finish reading shared memory before next TMA load
        __syncthreads();
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