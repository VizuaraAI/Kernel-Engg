/**
 * Simple, self-contained CUDA C++ implementation of FlashAttention-3 (Forward Pass)
 * demonstrating:
 *   1. Producer-Consumer Warp Specialization (Producer warp loads K/V tiles
 *      asynchronously into a 2-stage circular SMEM buffer; Consumer warps compute
 *      QK^T, online softmax, and PV).
 *   2. 2-Stage GEMM-Softmax Pipelining (Algorithm 2 from Shah et al., 2024):
 *      double-buffered score tiles S_cur / S_next so softmax and PV accumulation
 *      are decoupled across iterations.
 *
 * Compile & run:
 *   nvcc -O3 -std=c++17 lecture17/flash_attention_3.cu -o fa3 && ./fa3
 */

#include "../common/cuda_utils.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <vector>

constexpr int Br = 16;        // Query block size
constexpr int Bc = 16;        // Key/Value block size
constexpr int D_MAX = 64;     // Head dimension (compile-time bound for SMEM/RMEM)
constexpr int NUM_STAGES = 2; // Circular SMEM buffer stages

// 1 producer warp (32 threads) + 4 consumer warps (128 threads) = 160 threads
constexpr int PRODUCER_THREADS = 32;
constexpr int CONSUMER_THREADS = 128;
constexpr int BLOCK_THREADS = PRODUCER_THREADS + CONSUMER_THREADS;

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

__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;" ::"r"(CONSUMER_THREADS) : "memory"); }

__global__ void flash_attention_3_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O,
                                         int N, int d, float scale) {
    // One CTA per (batch, head, q_block)
    const int q_block = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const bool is_producer = (tid < PRODUCER_THREADS);

    const float* Q_bh = Q + bh * N * d;
    const float* K_bh = K + bh * N * d;
    const float* V_bh = V + bh * N * d;
    float* O_bh = O + bh * N * d;

    // Shared memory: Q_i + 2-stage circular buffers for K and V + score buffer
    __shared__ float sQ[Br][D_MAX];
    __shared__ float sK[NUM_STAGES][Bc][D_MAX];
    __shared__ float sV[NUM_STAGES][Bc][D_MAX];
    __shared__ float sS[2][Br][Bc]; // S_cur (0) and S_next (1) for 2-stage pipeline
    __shared__ float sP[Br][Bc];    // Softmax probabilities P_tilde

    // Producer-Consumer barriers for circular SMEM stages
    using Barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ Barrier bar_full[NUM_STAGES];
    __shared__ Barrier bar_empty[NUM_STAGES];
    __shared__ Barrier bar_q;

    if (tid == 0) {
        init(&bar_q, BLOCK_THREADS);
        for (int s = 0; s < NUM_STAGES; ++s) {
            init(&bar_full[s], PRODUCER_THREADS + CONSUMER_THREADS);
            init(&bar_empty[s], PRODUCER_THREADS + CONSUMER_THREADS);
        }
    }
    __syncthreads();

    const int num_kv_blocks = (N + Bc - 1) / Bc;
    const int max_kv_blocks = min(num_kv_blocks, ((q_block + 1) * Br + Bc - 1) / Bc);

    // =========================================================================
    // PRODUCER WARP: Asynchronously stream Q_i, K_j, V_j into circular SMEM
    // =========================================================================
    if (is_producer) {
        // Load Q_i once
        for (int idx = tid; idx < Br * d; idx += PRODUCER_THREADS) {
            int r = idx / d;
            int c = idx % d;
            int gr = q_block * Br + r;
            sQ[r][c] = (gr < N) ? Q_bh[gr * d + c] : 0.0f;
        }
        bar_q.arrive_and_wait();

        for (int j = 0; j < max_kv_blocks; ++j) {
            int stage = j % NUM_STAGES;
            if (j >= NUM_STAGES) {
                // Wait for consumer to finish with this stage before overwriting
                bar_empty[stage].arrive_and_wait();
            }
            for (int idx = tid; idx < Bc * d; idx += PRODUCER_THREADS) {
                int r = idx / d;
                int c = idx % d;
                int gr = j * Bc + r;
                sK[stage][r][c] = (gr < N) ? K_bh[gr * d + c] : 0.0f;
                sV[stage][r][c] = (gr < N) ? V_bh[gr * d + c] : 0.0f;
            }
            bar_full[stage].arrive_and_wait();
        }
        return;
    }

    // =========================================================================
    // CONSUMER WARPS: 2-Stage GEMM-Softmax Pipeline (Algorithm 2)
    // =========================================================================
    const int ctid = tid - PRODUCER_THREADS; // 0 .. 127
    const int warp_id = ctid / 32;           // 0 .. 3
    const int lane_id = ctid % 32;           // 0 .. 31

    // Wait for Q_i
    bar_q.arrive_and_wait();

    if (max_kv_blocks == 0)
        return;

    // Each consumer warp handles a subset of rows of Br (Br=32 -> 8 rows/warp)
    // For simplicity, each thread in consumer warps owns rows via strided loop.
    float m_row[Br];
    float l_row[Br];
    float o_acc[Br][D_MAX];
    for (int r = 0; r < Br; ++r) {
        m_row[r] = -INFINITY;
        l_row[r] = 0.0f;
        for (int c = 0; c < d; ++c) {
            o_acc[r][c] = 0.0f;
        }
    }

    auto compute_qk = [&](int stage, int kv_block, int s_buf) {
        for (int idx = ctid; idx < Br * Bc; idx += CONSUMER_THREADS) {
            int r = idx / Bc;
            int c = idx % Bc;
            int gr = q_block * Br + r;
            int gc = kv_block * Bc + c;
            if (gc >= N || (gc > gr)) {
                sS[s_buf][r][c] = -INFINITY;
            } else {
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += sQ[r][k] * sK[stage][c][k];
                }
                sS[s_buf][r][c] = dot * scale;
            }
        }
    };

    auto compute_softmax = [&](int s_buf, float* scale_o_out) {
        // One warp per 8 rows (warp_id 0..3 handles rows warp_id*8 .. +7)
        for (int r = warp_id; r < Br; r += 4) {
            float val = (lane_id < Bc) ? sS[s_buf][r][lane_id] : -INFINITY;
            float row_m = warp_reduce_max(val);
            float m_old = m_row[r];
            float m_new = fmaxf(m_old, row_m);
            float alpha = (m_old > -INFINITY) ? expf(m_old - m_new) : 0.0f;

            float p_val = (lane_id < Bc && val > -INFINITY) ? expf(val - m_new) : 0.0f;
            float row_l = warp_reduce_sum(p_val);

            if (lane_id < Bc) {
                sP[r][lane_id] = p_val;
            }
            m_row[r] = m_new;
            l_row[r] = l_row[r] * alpha + row_l;
            scale_o_out[r] = alpha;
        }
    };

    auto accumulate_pv = [&](int stage) {
        // Each consumer warp updates its assigned rows (warp_id, warp_id+4, ...)
        for (int r = warp_id; r < Br; r += 4) {
            for (int c = lane_id; c < d; c += 32) {
                float pv = 0.0f;
                for (int k = 0; k < Bc; ++k) {
                    pv += sP[r][k] * sV[stage][k][c];
                }
                o_acc[r][c] += pv;
            }
        }
    };

    float scale_o[Br];

    // Prologue: wait K_0, compute S_cur = Q_i @ K_0^T, softmax(S_cur)
    bar_full[0].arrive_and_wait();
    compute_qk(0, 0, 0);
    consumer_sync();
    compute_softmax(0, scale_o);
    consumer_sync();

    // Mainloop: 2-stage pipeline (Algorithm 2)
    int cur_buf = 0;
    for (int j = 1; j < max_kv_blocks; ++j) {
        int next_buf = 1 - cur_buf;
        int k_stage = j % NUM_STAGES;
        int v_stage = (j - 1) % NUM_STAGES;

        // Wait for K_j to arrive in SMEM
        bar_full[k_stage].arrive_and_wait();

        // 1. Compute S_next = Q_i @ K_j^T
        compute_qk(k_stage, j, next_buf);
        // 2. Accumulate O_i += P_cur @ V_{j-1}
        accumulate_pv(v_stage);
        consumer_sync();

        // 3. Softmax on S_next -> P_next
        compute_softmax(next_buf, scale_o);
        consumer_sync();

        // 4. Rescale O_i by exp(m_old - m_new)
        for (int r = warp_id; r < Br; r += 4) {
            float alpha = scale_o[r];
            for (int c = lane_id; c < d; c += 32) {
                o_acc[r][c] *= alpha;
            }
        }

        // Release V_{j-1} stage back to producer
        (void)bar_empty[v_stage].arrive();
        cur_buf = next_buf;
    }

    // Tail: accumulate P_last @ V_{last}
    int last_stage = (max_kv_blocks - 1) % NUM_STAGES;
    accumulate_pv(last_stage);
    (void)bar_empty[last_stage].arrive();

    // Epilogue: normalize O_i by l_i and write to HBM
    for (int r = warp_id; r < Br; r += 4) {
        int gr = q_block * Br + r;
        if (gr < N) {
            float inv_l = 1.0f / l_row[r];
            for (int c = lane_id; c < d; c += 32) {
                O_bh[gr * d + c] = o_acc[r][c] * inv_l;
            }
        }
    }
}

void cpu_attention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& O, int BH, int N, int d,
                   float scale) {
    for (int bh = 0; bh < BH; ++bh) {
        const float* q = Q.data() + bh * N * d;
        const float* k = K.data() + bh * N * d;
        const float* v = V.data() + bh * N * d;
        float* o = O.data() + bh * N * d;

        std::vector<float> scores(N);
        for (int i = 0; i < N; ++i) {
            float max_s = -INFINITY;
            for (int j = 0; j < N; ++j) {
                if (j > i) {
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
    const int B = 1;
    const int H = 4;
    const int N = 256;
    const int d = 64;
    const int BH = B * H;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    size_t bytes = static_cast<size_t>(BH) * N * d * sizeof(float);
    std::vector<float> hostQ(BH * N * d);
    std::vector<float> hostK(BH * N * d);
    std::vector<float> hostV(BH * N * d);
    std::vector hostO(BH * N * d, 0.0f);
    std::vector h_ref(BH * N * d, 0.0f);

    srand(0);
    for (size_t i = 0; i < hostQ.size(); ++i) {
        hostQ[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        hostK[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        hostV[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    float* d_Q;
    float* d_K;
    float* d_V;
    float* d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, hostQ.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, hostK.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, hostV.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid((N + Br - 1) / Br, BH);
    dim3 block(BLOCK_THREADS);

    flash_attention_3_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hostO.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    cpu_attention(hostQ, hostK, hostV, h_ref, BH, N, d, scale);

    float max_err = 0.0f;
    for (size_t i = 0; i < hostO.size(); ++i) {
        max_err = fmaxf(max_err, fabsf(hostO[i] - h_ref[i]));
    }
    printf("FlashAttention-3 CUDA C++ max absolute error: %e\n", max_err);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return (max_err < 1e-4f) ? 0 : 1;
}
