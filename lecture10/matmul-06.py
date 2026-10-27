# %%writefile matmul-06.py
from numba import cuda, float32
import cupy as cp
import numpy as np

ceil_div = lambda a, b: (a + b - 1) // b

WARP_SIZE = 32

# Block tile dimensions and chunk width
BM = 64
BN = 64
BK = 16

# Thread patch dimensions (each thread owns one TM x TN patch = 4 x 4 = 16 elements)
TM = 4
TN = 4

# Warp tile dimensions: 32 threads * (4 x 4) = 512 elements -> 16 x 32 region per warp
# (4 patches tall by 8 patches wide = 32 patches per warp)
WM = 16
WN = 32

# Total threads per block: (64 / 4) * (64 / 4) = 256 threads (8 warps)
NT = (BM // TM) * (BN // TN)

LOAD_SIZE = 4

@cuda.jit
def load_tileA(tileA, matA, tid, bk_offset, block_row):
    for i in range(tid, (BM * BK) // 4, NT):
        loads_per_row = BK // 4
        load_grid_row = i // loads_per_row
        load_grid_col = i % loads_per_row
        load_row = load_grid_row * 1
        load_col = load_grid_col * 4
        tileA[load_col + 0, load_row] = matA[block_row + load_row, bk_offset + load_col + 0]
        tileA[load_col + 1, load_row] = matA[block_row + load_row, bk_offset + load_col + 1]
        tileA[load_col + 2, load_row] = matA[block_row + load_row, bk_offset + load_col + 2]
        tileA[load_col + 3, load_row] = matA[block_row + load_row, bk_offset + load_col + 3]

@cuda.jit
def load_tileB(tileB, matB, tid, bk_offset, block_col):
    for i in range(tid, (BK * BN) // 4, NT):
        loads_per_row = BN // 4
        load_grid_row = i // loads_per_row
        load_grid_col = i % loads_per_row
        load_row = load_grid_row * 1
        load_col = load_grid_col * 4
        tileB[load_row, load_col + 0] = matB[bk_offset + load_row, block_col + load_col + 0]
        tileB[load_row, load_col + 1] = matB[bk_offset + load_row, block_col + load_col + 1]
        tileB[load_row, load_col + 2] = matB[bk_offset + load_row, block_col + load_col + 2]
        tileB[load_row, load_col + 3 ] = matB[bk_offset + load_row, block_col + load_col + 3]
        

@cuda.jit
def matmul(matA, matB, matC, m, n, k):
    tileA = cuda.shared.array(shape=(BK, BM), dtype=cp.float32)
    tileB = cuda.shared.array(shape=(BK, BN), dtype=cp.float32)
    
    tid = cuda.threadIdx.x
    
    warp_id = tid // WARP_SIZE
    lane_id = tid % WARP_SIZE
    
    warps_per_row = BN // WN
    warp_grid_row = warp_id // warps_per_row    # w.r.t. warp grid
    warp_grid_col = warp_id % warps_per_row     # w.r.t. warp grid
    warp_row = warp_grid_row * WM               # w.r.t. thread block
    warp_col = warp_grid_col * WN               # w.r.t. thread block
    
    # Given tid
    # which (WM, WN) region it maps to
    # which (TM, TN) patch does it map to
    patches_per_warp_tile = WN // TN
    patch_row_in_warp = lane_id // patches_per_warp_tile
    patch_col_in_warp = lane_id % patches_per_warp_tile
    patch_row = patch_row_in_warp * TM
    patch_col = patch_col_in_warp * TN
    
    thread_row = warp_row + patch_row       # w.r.t. thread block
    thread_col = warp_col + patch_col       # w.r.t. thread block
    
    block_row = cuda.blockIdx.y * BM        # position of block w.r.t the grid
    block_col = cuda.blockIdx.x * BN        # position of block w.r.t the grid
    
    # register allocation
    product = cuda.local.array(shape=(TM, TN), dtype=cp.float32)
    regM = cuda.local.array(TM, dtype=cp.float32)
    regN = cuda.local.array(TN, dtype=cp.float32)
    
    for bk_offset in range(0, k, BK):
        # phase 1a: load from tileA
        load_tileA(tileA, matA, tid, bk_offset, block_row)
        load_tileB(tileB, matB, tid, bk_offset, block_col)
        
        cuda.syncthreads()
        
        # phase 2
        for dot_k in range(BK):
            for i in range(0, TM, LOAD_SIZE):
                regM[i + 0] = tileA[dot_k, thread_row + i + 0]
                regM[i + 1] = tileA[dot_k, thread_row + i + 1]
                regM[i + 2] = tileA[dot_k, thread_row + i + 2]
                regM[i + 3] = tileA[dot_k, thread_row + i + 3]
            for j in range(0, TN, LOAD_SIZE):
                regN[j + 0] = tileB[dot_k, thread_col + j + 0]
                regN[j + 1] = tileB[dot_k, thread_col + j + 1]
                regN[j + 2] = tileB[dot_k, thread_col + j + 2]
                regN[j + 3] = tileB[dot_k, thread_col + j + 3]
                
            for i in range(TM):
                for j in range(TN):
                    product[i, j] += regM[i] * regN[j]
    
        cuda.syncthreads()
        
    # write register results back to global memory
    for i in range(TM):
        for j in range(TN):
            abs_row_offset = block_row + thread_row
            abs_col_offset = block_col + thread_col
            matC[abs_row_offset + i, abs_col_offset + j] = product[i, j]

m = 1024
k = 1024
n = 1024
matA = cp.ones(shape=(m, k), dtype=cp.float32) * 2.0
matB = cp.ones(shape=(k, n), dtype=cp.float32) * 3.0
matC = cp.zeros(shape=(m, n), dtype=cp.float32)

threads_per_block = (NT, 1)
blocks_per_grid = (
    ceil_div(n, BN),
    ceil_div(m, BM),
)
matmul[blocks_per_grid, threads_per_block](matA, matB, matC, m, n, k)
cuda.synchronize()

matA_np = cp.asnumpy(matA)
matB_np = cp.asnumpy(matB)
matC_np = cp.asnumpy(matC)

np.testing.assert_allclose(matC_np, matA_np @ matB_np, rtol=1e-5, atol=1e-5)