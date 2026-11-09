#include <iostream>
#include <cuda_runtime.h>

// Problem: 2D Matrix Transpose using Shared Memory Tiles
// Dimensions: 2048 x 2048
#define WIDTH 2048
#define HEIGHT 2048
#define TILE_DIM 32
#define BLOCK_ROWS 8

// =============================================================================
// KERNEL 2 (FIXED): Zero Bank Conflicts via Padding
//
// NVIDIA Triage Path: Decision 2 -> Decision 3 & Decision 5 -> Decision 6
//
// The Fix:
//   Pad the tile stride by 1 float: __shared__ float tile[TILE_DIM][TILE_DIM + 1];
//   Bank index = (i * 33 + 0) % 32 = i % 32.
//   Thread i in the warp now accesses Bank i. Every thread accesses a distinct bank!
//
// Baseline Difference in NCU:
//   - Bank conflicts drop from millions to ZERO (-100%)
//   - Shared memory wavefronts per request drop from 32 to 1 (-97%)
//   - "Stall MIO Throttle" collapses to near zero
//   - Kernel duration improves by ~3x to 5x
// =============================================================================
__global__ void transpose_fixed(const float* in, float* out, int width, int height) {
    // FIX: Pad row stride to 33 floats to eliminate bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

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

    // FIX: Conflict-free column read from padded tile
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

    // Launch kernel once
    transpose_fixed<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    std::cout << "[Kernel 2 FIXED] Transposed out[1 * HEIGHT + 0]: " << h_out[1 * HEIGHT + 0] 
              << " (Expected: " << h_in[0 * WIDTH + 1] << ")\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
