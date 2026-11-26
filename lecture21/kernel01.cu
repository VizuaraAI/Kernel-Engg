#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "../common/cuda_utils.cuh"

constexpr int WARP_SIZE = 32;

// Block tile dimensions and chunk width
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

// Thread patch dimensions (each thread owns a TM x TN = 4 x 4 patch)
constexpr int TM = 4;
constexpr int TN = 4;

// Warp tile dimensions: 32 threads * (4 x 4) = 512 elements -> 16 x 32 region per warp
constexpr int WM = 16;
constexpr int WN = 32;

// Threads per block: (64/4) * (64/4) = 256 (8 warps)
constexpr int NT = (BM / TM) * (BN / TN);

constexpr int LOAD_SIZE = 4;

// Load a BM x BK tile of A (row-major, leading dim k) into shared memory,
// transposed so that tileA is [BK][BM].
__device__ __forceinline__ void load_tileA(float (*tileA)[BM], const float* matA, int tid,
                                           int bk_offset, int block_row, int k) {
    constexpr int loads_per_row = BK / 4;
    for (int i = tid; i < (BM * BK) / 4; i += NT) {
        int load_row = i / loads_per_row;
        int load_col = (i % loads_per_row) * 4;

        float4 v = *reinterpret_cast<const float4*>(
            &matA[(size_t)(block_row + load_row) * k + bk_offset + load_col]);

        tileA[load_col + 0][load_row] = v.x;
        tileA[load_col + 1][load_row] = v.y;
        tileA[load_col + 2][load_row] = v.z;
        tileA[load_col + 3][load_row] = v.w;
    }
}

// Load a BK x BN tile of B (row-major, leading dim n) into shared memory, [BK][BN].
__device__ __forceinline__ void load_tileB(float (*tileB)[BN], const float* matB, int tid,
                                           int bk_offset, int block_col, int n) {
    constexpr int loads_per_row = BN / 4;
    for (int i = tid; i < (BK * BN) / 4; i += NT) {
        int load_row = i / loads_per_row;
        int load_col = (i % loads_per_row) * 4;

        float4 v = *reinterpret_cast<const float4*>(
            &matB[(size_t)(bk_offset + load_row) * n + block_col + load_col]);

        *reinterpret_cast<float4*>(&tileB[load_row][load_col]) = v;
    }
}

__global__ void __launch_bounds__(NT)
    matmul(const float* __restrict__ matA, const float* __restrict__ matB, float* __restrict__ matC,
           int m, int n, int k) {
    __shared__ __align__(16) float tileA[BK][BM];
    __shared__ __align__(16) float tileB[BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    constexpr int warps_per_row = BN / WN;
    const int warp_grid_row = warp_id / warps_per_row; // w.r.t. warp grid
    const int warp_grid_col = warp_id % warps_per_row;
    const int warp_row = warp_grid_row * WM; // w.r.t. thread block
    const int warp_col = warp_grid_col * WN;

    // Which (TM x TN) patch inside the warp tile this lane owns
    constexpr int patches_per_warp_tile = WN / TN;
    const int patch_row_in_warp = lane_id / patches_per_warp_tile;
    const int patch_col_in_warp = lane_id % patches_per_warp_tile;
    const int patch_row = patch_row_in_warp * TM;
    const int patch_col = patch_col_in_warp * TN;

    const int thread_row = warp_row + patch_row; // w.r.t. thread block
    const int thread_col = warp_col + patch_col;

    const int block_row = blockIdx.y * BM; // w.r.t. grid
    const int block_col = blockIdx.x * BN;

    // Register allocation
    float product[TM][TN] = {}; // zero-initialised
    float regM[TM];
    float regN[TN];

    for (int bk_offset = 0; bk_offset < k; bk_offset += BK) {
        // Phase 1: global -> shared
        load_tileA(tileA, matA, tid, bk_offset, block_row, k);
        load_tileB(tileB, matB, tid, bk_offset, block_col, n);

        __syncthreads();

// Phase 2: shared -> registers -> FMA
#pragma unroll
        for (int dot_k = 0; dot_k < BK; ++dot_k) {
#pragma unroll
            for (int i = 0; i < TM; i += LOAD_SIZE) {
                float4 v = *reinterpret_cast<const float4*>(&tileA[dot_k][thread_row + i]);
                regM[i + 0] = v.x;
                regM[i + 1] = v.y;
                regM[i + 2] = v.z;
                regM[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += LOAD_SIZE) {
                float4 v = *reinterpret_cast<const float4*>(&tileB[dot_k][thread_col + j]);
                regN[j + 0] = v.x;
                regN[j + 1] = v.y;
                regN[j + 2] = v.z;
                regN[j + 3] = v.w;
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    product[i][j] += regM[i] * regN[j];
                }
            }
        }

        __syncthreads();
    }

    // Write register results back to global memory (float4 per row of the patch)
    const int abs_row = block_row + thread_row;
    const int abs_col = block_col + thread_col;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        float4 v = make_float4(product[i][0], product[i][1], product[i][2], product[i][3]);
        *reinterpret_cast<float4*>(&matC[(size_t)(abs_row + i) * n + abs_col]) = v;
    }
}

int main() {
    // NOTE: kernel assumes m % BM == 0, n % BN == 0, k % BK == 0 (no bounds checks)
    const int m = 4096, k = 4096, n = 4096;

    std::vector<float> hA((size_t)m * k, 2.0f);
    std::vector<float> hB((size_t)k * n, 3.0f);
    std::vector<float> hC((size_t)m * n, 0.0f);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
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