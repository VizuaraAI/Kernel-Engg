# %%writefile matmul-05.py
from numba import cuda, float32
import cupy as cp
import numpy as np

ceil_div = lambda a, b: (a + b - 1) // b

# Block tile dimensions (BM x BN) and chunk width (BK)
BM = 64
BN = 64
BK = 16

# Thread patch dimensions (TM x TN) owned by each thread in registers
TM = 4
TN = 4

# Number of threads per block: (64 / 4) * (64 / 4) = 16 * 16 = 256
NT = (BM // TM) * (BN // TN)

@cuda.jit
def load_tileA(tileA, matA, tid, bk_offset, block_row):
    for i in range(tid, BM * BK, NT):
        r = i // BK
        c = i % BK
        tileA[r, c] = matA[block_row + r, bk_offset + c]


@cuda.jit
def load_tileB(tileB, matB, tid, bk_offset, block_col):
    for i in range(tid, BK * BN, NT):
        r = i // BN
        c = i % BN
        tileB[r, c] = matB[bk_offset + r, block_col + c]


@cuda.jit
def matmul(matA, matB, matC, m, n, k):
    tileA = cuda.shared.array(shape=(BM, BK), dtype=cp.float32)
    tileB = cuda.shared.array(shape=(BK, BN), dtype=cp.float32)
    
    tid = cuda.threadIdx.x
    patches_per_row = BN // TN
    patch_grid_row = tid // patches_per_row
    patch_grid_col = tid % patches_per_row
    patch_row = patch_grid_row * TM
    patch_col = patch_grid_col * TN
    
    block_row = cuda.blockIdx.y * BM
    block_col = cuda.blockIdx.x * BN
    
    # register allocation
    product = cuda.local.array(shape=(TM, TN), dtype=cp.float32)
    regM = cuda.local.array(TM, dtype=cp.float32)
    regN = cuda.local.array(TN, dtype=cp.float32)
    
    for bk_offset in range(0, k, BK):
        # phase 1: all threads populating tileA and tileB
        # gmem -> smem
        load_tileA(tileA, matA, tid, bk_offset, block_row)
        load_tileB(tileB, matB, tid, bk_offset, block_col)
        
        cuda.syncthreads()
        
        # phase 2: computing TM * TN outer product  
        for dot_k in range(BK):
            for i in range(TM):
                regM[i] = tileA[patch_row + i, dot_k]
            for j in range(TN):
                regN[j] = tileB[dot_k, patch_col + j]
                
            # compute the outer product
            for i in range(TM):
                for j in range(TN):
                    product[i, j] += regM[i] * regN[j]
            
        cuda.syncthreads()
        
    # write back results from registers to matC (gmem)
    for i in range(TM):
        for j in range(TN):
            abs_row_offset = block_row + patch_row
            abs_col_offset = block_col + patch_col
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