#include <iostream>
#include <cuda_runtime.h>

// Problem: 2D Matrix Transpose using Shared Memory Tiles
// Dimensions: 2048 x 2048
#define WIDTH 2048
#define HEIGHT 2048
#define TILE_DIM 32
#define BLOCK_ROWS 8

// =============================================================================
// KERNEL 2 (SABOTAGED): 32-Way Shared Memory Bank Conflicts
//
// NVIDIA Triage Path: Decision 2 -> Decision 3 & Decision 5 -> Decision 6
//
// Root Cause:
//   Unpadded 32x32 shared memory tile: __shared__ float tile[TILE_DIM][TILE_DIM];
//   When reading column-wise: tile[threadIdx.x][threadIdx.y + j]
//   Within warp 0, thread i accesses: tile[i][0]
//   Bank index = (i * 32 + 0) % 32 = 0.
//   All 32 threads collide on Bank 0 -> 32 serial wavefronts!
//
// NCU Diagnostic Signatures:
//   - Speed of Light: High occupancy (>80%), but low issue rate (<0.2)
//   - Warp State Statistics: Dominated by "Stall MIO Throttle" and "Stall Short Scoreboard"
//   - Memory Workload Analysis (Shared):
//       * Millions of bank conflicts (l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum)
//       * Shared memory wavefronts per request = 32 (ideal is 1)
//   - Source Page: LDS (Load Shared) instruction carries the MIO Throttle stall samples
// =============================================================================
__global__ void transpose_sabotaged(const float* in, float* out, int width, int height) {
    // SABOTAGE: Unpadded 32x32 array -> causes 32-way bank conflict on column reads!
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced global load into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }

    __syncthreads();

    // Transpose tile coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // SABOTAGE: Column read from tile -> 32-way bank conflict across warp!
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main() {
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    float *h_in = new float[WIDTH * HEIGHT];
    float *h_out = new float[WIDTH * HEIGHT];
    for (int i = 0; i < WIDTH * HEIGHT; ++i) h_in[i] = static_cast<float>(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(WIDTH / TILE_DIM, HEIGHT / TILE_DIM);

    // Warm-up launch
    transpose_sabotaged<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Timed launch
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 20; ++i) {
        transpose_sabotaged<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "[Kernel 2 SABOTAGED] Average Duration: " << (ms * 50.0f) << " us\n";

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Sample transposed out[1 * HEIGHT + 0]: " << h_out[1 * HEIGHT + 0] 
              << " (Expected: " << h_in[0 * WIDTH + 1] << ")\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
