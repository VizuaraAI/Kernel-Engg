/**
 * Simple, self-contained CUDA C++ (SIMT) implementation of FlashAttention-4 (Forward Pass)
 * modeled after Zadouri et al. (2026), "FlashAttention-4: Algorithm and Kernel Pipelining
 * Co-Design for Asymmetric Hardware Scaling", adapted to CUDA SIMT following flash_attention_3.cu:
 *
 *   1. Ping-Pong 2-Tile Q Schedule & 4-Role Warp Specialization:
 *      - Producer warp (1 warp): asynchronously streams Q^H, Q^L and K_j, V_j into a
 *        2-stage circular SMEM buffer.
 *      - Softmax-High & Softmax-Low warpgroups (2 warps each): compute QK^T and
 *        ping-pong their exponential critical sections via explicit barriers so they
 *        never overlap in exponential compute.
 *      - Correction warp (1 warp): decoupled from softmax critical path; reads P
 *        and rescale factors from SMEM (modeling Blackwell TMEM), rescales O, and
 *        accumulates P * V.
 *   2. Software-Emulated Exponential (2^x) with Partial Emulation (Section 3.1.3):
 *      - Range reduction via IEEE-754 floor + degree-5 minimax polynomial on [0, 1)
 *        using Horner's method with fmaf() + integer exponent-bit shift.
 *      - Partial emulation (25% of elements via FMA emulation, 75% via hardware exp2f)
 *        to balance FMA and MUFU throughput.
 *   3. Conditional Softmax Rescaling (Section 3.1.4):
 *      - Skips rescaling O and l when (m_new - m_old) <= TAU (TAU = log2(256) = 8.0),
 *        only rescaling when the running max jumps by more than TAU.
 *   4. LPT (Longest-Processing-Time-First) Causal Scheduling (Section 3.3):
 *      - Reverses query-tile block order when is_causal=true so longest mainloops
 *        execute first, minimizing tail-effect load imbalance.
 *
 * Compile & run:
 *   nvcc -O3 -std=c++17 lecture17/flash_attention_4.cu -o fa4 && ./fa4
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call)                                                                                                                             \
    do {                                                                                                                                             \
        cudaError_t err = (call);                                                                                                                    \
        if (err != cudaSuccess) {                                                                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                                               \
            exit(EXIT_FAILURE);                                                                                                                      \
        }                                                                                                                                            \
    } while (0)

constexpr int Br = 16;        // Query sub-tile size (2 * Br = 32 rows per CTA: High + Low)
constexpr int Bc = 16;        // Key/Value block size
constexpr int D_MAX = 64;     // Head dimension (compile-time bound for SMEM)
constexpr int NUM_STAGES = 2; // Circular SMEM buffer stages

// Warp specialization:
//   Warp 0      : Producer (32 threads)
//   Warps 1..2  : Softmax-High (64 threads)
//   Warps 3..4  : Softmax-Low  (64 threads)
//   Warp 5      : Correction   (32 threads)
constexpr int PRODUCER_THREADS = 32;
constexpr int SOFTMAX_H_THREADS = 64;
constexpr int SOFTMAX_L_THREADS = 64;
constexpr int CORRECTION_THREADS = 32;
constexpr int BLOCK_THREADS = PRODUCER_THREADS + SOFTMAX_H_THREADS + SOFTMAX_L_THREADS + CORRECTION_THREADS;

constexpr float LOG2_E = 1.4426950408889634f;
constexpr float TAU_LOG2 = 8.0f; // Conditional rescaling threshold: log2(256) = 8.0

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Software-emulated 2^x using Cody-Waite / IEEE-754 range reduction and
 * degree-5 minimax polynomial approximation on [0, 1) via Horner's method (FMA).
 * Max relative error vs FP64: ~8.4e-8 (matches hardware MUFU.EX2 to < 2x).
 */
__device__ __forceinline__ float exp2_emu(float x) {
    if (x <= -126.0f)
        return 0.0f;

    // Step 1-2: Compute floor(x) using round-down IEEE-754 magic constant (2^23 + 2^22)
    constexpr float MAGIC = 12582912.0f; // 2^23 + 2^22
    float x_rd = __fadd_rd(x, MAGIC);
    float x_floor = x_rd - MAGIC;
    int32_t int_floor = static_cast<int32_t>(x_floor);

    // Step 3: Fractional part x_frac in [0, 1)
    float xf = x - x_floor;

    // Step 4: Degree-5 minimax polynomial on [0, 1) evaluated with Horner's method
    float p = 0.001867182864132132f;
    p = fmaf(p, xf, 0.009016687624481275f);
    p = fmaf(p, xf, 0.0558004470615143f);
    p = fmaf(p, xf, 0.24016415351263767f);
    p = fmaf(p, xf, 0.6931513628656082f);
    p = fmaf(p, xf, 1.0f);

    // Step 5: Combine 2^int_floor * 2^xf by shifting int_floor into IEEE-754 exponent bits
    uint32_t bits = __float_as_uint(p);
    bits += static_cast<uint32_t>(int_floor) << 23;
    return __uint_as_float(bits);
}

__global__ void flash_attention_4_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O,
                                         int N, int d, float scale, bool is_causal) {
    // LPT Scheduling (Section 3.3): for causal masking, reverse query-pair order
    // so CTAs with longest mainloops execute first.
    const int num_q_pairs = gridDim.x;
    const int q_pair = is_causal ? (num_q_pairs - 1 - blockIdx.x) : blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;

    const int q_block_H = q_pair * 2;
    const int q_block_L = q_pair * 2 + 1;

    const float* Q_bh = Q + bh * N * d;
    const float* K_bh = K + bh * N * d;
    const float* V_bh = V + bh * N * d;
    float* O_bh = O + bh * N * d;

    const float scale_log2 = scale * LOG2_E;

    // Shared memory:
    //   - sQ[2][Br][D_MAX]: High (0) and Low (1) query tiles
    //   - sK, sV: 2-stage circular buffers
    //   - sSP[2][Br][Bc]: Overlapping S and P tiles (High=0, Low=1)
    //   - sO[2][Br][D_MAX]: Output accumulators (modeling Blackwell TMEM)
    //   - sScale[2][Br], sL[2][Br]: Rescale factors and normalizers for Correction warp
    __shared__ float sQ[2][Br][D_MAX];
    __shared__ float sK[NUM_STAGES][Bc][D_MAX];
    __shared__ float sV[NUM_STAGES][Bc][D_MAX];
    __shared__ float sSP[2][Br][Bc];
    __shared__ float sO[2][Br][D_MAX];
    __shared__ float sScale[2][Br];
    __shared__ float sL[2][Br];

    using Barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ Barrier bar_q;
    __shared__ Barrier bar_k_full[NUM_STAGES];
    __shared__ Barrier bar_v_empty[NUM_STAGES];
    __shared__ Barrier bar_ping_pong[2];
    __shared__ Barrier bar_p_ready[2];
    __shared__ Barrier bar_p_empty[2];

    if (tid == 0) {
        init(&bar_q, BLOCK_THREADS);
        for (int s = 0; s < NUM_STAGES; ++s) {
            init(&bar_k_full[s], PRODUCER_THREADS + SOFTMAX_H_THREADS + SOFTMAX_L_THREADS);
            init(&bar_v_empty[s], PRODUCER_THREADS + CORRECTION_THREADS);
        }
        init(&bar_ping_pong[0], SOFTMAX_H_THREADS + SOFTMAX_L_THREADS);
        init(&bar_ping_pong[1], SOFTMAX_H_THREADS + SOFTMAX_L_THREADS);
        init(&bar_p_ready[0], SOFTMAX_H_THREADS + CORRECTION_THREADS);
        init(&bar_p_ready[1], SOFTMAX_L_THREADS + CORRECTION_THREADS);
        init(&bar_p_empty[0], SOFTMAX_H_THREADS + CORRECTION_THREADS);
        init(&bar_p_empty[1], SOFTMAX_L_THREADS + CORRECTION_THREADS);
    }
    __syncthreads();

    const int num_kv_blocks = (N + Bc - 1) / Bc;
    const int max_gr = min(N - 1, q_pair * 2 * Br + 2 * Br - 1);
    const int max_kv_blocks = is_causal ? min(num_kv_blocks, (max_gr + 1 + Bc - 1) / Bc) : num_kv_blocks;

    // =========================================================================
    // ROLE 1: PRODUCER WARP (threads 0 .. 31)
    // =========================================================================
    if (tid < PRODUCER_THREADS) {
        for (int t = 0; t < 2; ++t) {
            int qb = (t == 0) ? q_block_H : q_block_L;
            for (int idx = tid; idx < Br * d; idx += PRODUCER_THREADS) {
                int r = idx / d;
                int c = idx % d;
                int gr = qb * Br + r;
                sQ[t][r][c] = (gr < N) ? Q_bh[gr * d + c] : 0.0f;
            }
        }
        bar_q.arrive_and_wait();

        for (int j = 0; j < max_kv_blocks; ++j) {
            int stage = j % NUM_STAGES;
            if (j >= NUM_STAGES) {
                bar_v_empty[stage].arrive_and_wait();
            }
            for (int idx = tid; idx < Bc * d; idx += PRODUCER_THREADS) {
                int r = idx / d;
                int c = idx % d;
                int gr = j * Bc + r;
                sK[stage][r][c] = (gr < N) ? K_bh[gr * d + c] : 0.0f;
                sV[stage][r][c] = (gr < N) ? V_bh[gr * d + c] : 0.0f;
            }
            bar_k_full[stage].arrive_and_wait();
        }
        return;
    }

    // =========================================================================
    // ROLE 2 & 3: SOFTMAX-HIGH & SOFTMAX-LOW WARPGROUPS
    // =========================================================================
    if (tid < PRODUCER_THREADS + SOFTMAX_H_THREADS + SOFTMAX_L_THREADS) {
        const int tile_id = (tid < PRODUCER_THREADS + SOFTMAX_H_THREADS) ? 0 : 1;
        const int local_tid = (tile_id == 0) ? (tid - PRODUCER_THREADS) : (tid - PRODUCER_THREADS - SOFTMAX_H_THREADS);
        const int wg_threads = (tile_id == 0) ? SOFTMAX_H_THREADS : SOFTMAX_L_THREADS;
        const int warp_in_wg = local_tid / 32; // 0 .. 1
        const int lane_id = local_tid % 32;    // 0 .. 31
        const int q_block = (tile_id == 0) ? q_block_H : q_block_L;

        bar_q.arrive_and_wait();
        if (max_kv_blocks == 0)
            return;

        // Per-warpgroup online softmax state (warp_in_wg handles rows warp_in_wg, +2, ...)
        float m_row[Br];
        float l_row[Br];
        for (int r = 0; r < Br; ++r) {
            m_row[r] = -INFINITY;
            l_row[r] = 0.0f;
        }

        for (int j = 0; j < max_kv_blocks; ++j) {
            int stage = j % NUM_STAGES;
            bar_k_full[stage].arrive_and_wait();

            if (j > 0) {
                bar_p_empty[tile_id].arrive_and_wait();
            }

            // 1. Compute S_j = Q * K_j^T (pre-scaled by scale * log2(e))
            for (int idx = local_tid; idx < Br * Bc; idx += wg_threads) {
                int r = idx / Bc;
                int c = idx % Bc;
                int gr = q_block * Br + r;
                int gc = j * Bc + c;
                if (gr >= N || gc >= N || (is_causal && gc > gr)) {
                    sSP[tile_id][r][c] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int k = 0; k < d; ++k) {
                        dot += sQ[tile_id][r][k] * sK[stage][c][k];
                    }
                    sSP[tile_id][r][c] = dot * scale_log2;
                }
            }
            // Synchronize warps within this warpgroup before reading sSP
            asm volatile("bar.sync %0, %1;" ::"r"(tile_id + 1), "r"(wg_threads) : "memory");

            // Ping-pong synchronization: Softmax-Low waits for Softmax-High's exp section
            if (tile_id == 1) {
                bar_ping_pong[0].arrive_and_wait();
            }

            // 2. Softmax Critical Section (Partial Exp2 Emulation + Conditional Rescaling)
            for (int r = warp_in_wg; r < Br; r += 2) {
                float val = (lane_id < Bc) ? sSP[tile_id][r][lane_id] : -INFINITY;
                float block_m = warp_reduce_max(val);

                float m_old = m_row[r];
                float m_used = m_old;
                float alpha = 1.0f;

                if (block_m > -INFINITY) {
                    if (m_old == -INFINITY) {
                        m_used = block_m;
                        alpha = 0.0f;
                    } else if (block_m - m_old > TAU_LOG2) {
                        // Conditional rescaling triggered: max jumped by > TAU_LOG2
                        m_used = block_m;
                        alpha = exp2f(m_old - m_used);
                    }
                    // Otherwise skip updating m_row and set alpha = 1.0f (no O/l rescaling)
                } else {
                    alpha = (m_old == -INFINITY) ? 0.0f : 1.0f;
                }

                float p_val = 0.0f;
                if (lane_id < Bc && val > -INFINITY) {
                    float shifted = val - m_used;
                    // Partial emulation: 25% of entries via FMA polynomial, 75% via MUFU.EX2
                    if ((lane_id & 3) == 0) {
                        p_val = exp2_emu(shifted);
                    } else {
                        p_val = exp2f(shifted);
                    }
                }
                float row_l = warp_reduce_sum(p_val);

                if (lane_id < Bc) {
                    sSP[tile_id][r][lane_id] = p_val;
                }
                m_row[r] = m_used;
                l_row[r] = (alpha == 1.0f) ? (l_row[r] + row_l) : (l_row[r] * alpha + row_l);

                if (lane_id == 0) {
                    sScale[tile_id][r] = alpha;
                    if (j + 1 == max_kv_blocks) {
                        sL[tile_id][r] = l_row[r];
                    }
                }
            }

            // Ping-pong handoff
            if (tile_id == 0) {
                bar_ping_pong[0].arrive_and_wait();
                bar_p_ready[0].arrive_and_wait();
                bar_ping_pong[1].arrive_and_wait();
            } else {
                bar_ping_pong[1].arrive_and_wait();
                bar_p_ready[1].arrive_and_wait();
            }
        }
        return;
    }

    // =========================================================================
    // ROLE 4: CORRECTION WARP (threads 160 .. 191)
    // =========================================================================
    const int ctid = tid - (PRODUCER_THREADS + SOFTMAX_H_THREADS + SOFTMAX_L_THREADS);
    bar_q.arrive_and_wait();

    // Initialize sO in SMEM (modeling TMEM)
    for (int t = 0; t < 2; ++t) {
        for (int idx = ctid; idx < Br * d; idx += CORRECTION_THREADS) {
            sO[t][idx / d][idx % d] = 0.0f;
        }
    }
    __syncwarp();

    if (max_kv_blocks == 0)
        return;

    for (int j = 0; j < max_kv_blocks; ++j) {
        int stage = j % NUM_STAGES;

        // Process High tile (0), then Low tile (1) as P becomes ready
        for (int t = 0; t < 2; ++t) {
            bar_p_ready[t].arrive_and_wait();

            for (int idx = ctid; idx < Br * d; idx += CORRECTION_THREADS) {
                int r = idx / d;
                int c = idx % d;
                float alpha = sScale[t][r];

                // Conditional rescaling: skip multiplying sO when alpha == 1.0f
                float o_val = (alpha == 1.0f) ? sO[t][r][c] : (sO[t][r][c] * alpha);

                float pv = 0.0f;
                for (int k = 0; k < Bc; ++k) {
                    pv += sSP[t][r][k] * sV[stage][k][c];
                }
                sO[t][r][c] = o_val + pv;
            }
            __syncwarp();

            if (j + 1 < max_kv_blocks) {
                (void)bar_p_empty[t].arrive();
            }
        }

        if (j + NUM_STAGES < max_kv_blocks) {
            (void)bar_v_empty[stage].arrive();
        }
    }

    // Epilogue: normalize sO by sL and write to global memory
    for (int t = 0; t < 2; ++t) {
        int q_block = (t == 0) ? q_block_H : q_block_L;
        for (int idx = ctid; idx < Br * d; idx += CORRECTION_THREADS) {
            int r = idx / d;
            int c = idx % d;
            int gr = q_block * Br + r;
            if (gr < N) {
                O_bh[gr * d + c] = sO[t][r][c] / sL[t][r];
            }
        }
    }
}

void cpu_attention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& O, int BH, int N, int d,
                   float scale, bool is_causal) {
    for (int bh = 0; bh < BH; ++bh) {
        const float* q = Q.data() + bh * N * d;
        const float* k = K.data() + bh * N * d;
        const float* v = V.data() + bh * N * d;
        float* o = O.data() + bh * N * d;

        std::vector<float> scores(N);
        for (int i = 0; i < N; ++i) {
            float max_s = -INFINITY;
            for (int j = 0; j < N; ++j) {
                if (is_causal && j > i) {
                    scores[j] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int x = 0; x < d; ++x)
                        dot += q[i * d + x] * k[j * d + x];
                    scores[j] = dot * scale;
                }
                max_s = fmaxf(max_s, scores[j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < N; ++j) {
                scores[j] = (scores[j] > -INFINITY) ? expf(scores[j] - max_s) : 0.0f;
                sum_exp += scores[j];
            }
            for (int x = 0; x < d; ++x) {
                float acc = 0.0f;
                for (int j = 0; j < N; ++j)
                    acc += scores[j] * v[j * d + x];
                o[i * d + x] = acc / sum_exp;
            }
        }
    }
}

int main() {
    const int B = 1, H = 4, N = 256, d = 64;
    const int BH = B * H;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const bool is_causal = true;

    size_t bytes = static_cast<size_t>(BH) * N * d * sizeof(float);
    std::vector<float> h_Q(BH * N * d), h_K(BH * N * d), h_V(BH * N * d);
    std::vector<float> h_O(BH * N * d, 0.0f), h_ref(BH * N * d, 0.0f);

    srand(0);
    for (size_t i = 0; i < h_Q.size(); ++i) {
        h_Q[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid((N + 2 * Br - 1) / (2 * Br), BH);
    dim3 block(BLOCK_THREADS);

    flash_attention_4_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, scale, is_causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    cpu_attention(h_Q, h_K, h_V, h_ref, BH, N, d, scale, is_causal);

    float max_err = 0.0f;
    for (size_t i = 0; i < h_O.size(); ++i) {
        max_err = fmaxf(max_err, fabsf(h_O[i] - h_ref[i]));
    }
    printf("FlashAttention-4 CUDA C++ (SIMT) max absolute error: %e\n", max_err);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return (max_err < 1e-4f) ? 0 : 1;
}
