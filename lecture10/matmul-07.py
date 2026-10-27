from numba import cuda, float32
import cupy as cp
import numpy as np

ceil_div = lambda a, b: (a + b - 1) // b

WARP_SIZE = 32
MAX_THREADS_PER_BLOCK = 1024
MAX_REGISTERS_PER_THREAD = 255
MAX_SHARED_BYTES_PER_BLOCK = 48 * 1024  # 48 KB

def is_valid_config(BM, BN, BK, WM, WN, TM, TN):
    """Check whether a candidate (BM, BN, BK, WM, WN, TM, TN) obeys all GPU rules."""
    if BK % 4 != 0 or BN % 4 != 0 or TM % 4 != 0 or TN % 4 != 0:
        return False

    # Block tile must be evenly covered by warp tiles, and warp tiles by thread patches
    if BM % WM != 0 or BN % WN != 0 or WM % TM != 0 or WN % TN != 0:
        return False

    # Rule 1: Each warp's WM x WN region must hold exactly 32 patches of TM x TN
    if (WM // TM) * (WN // TN) != WARP_SIZE:
        return False

    # Rule 2: Threads in a block must make whole warps and number at most 1,024
    nt = (BM // TM) * (BN // TN)
    if nt % WARP_SIZE != 0 or nt < WARP_SIZE or nt > MAX_THREADS_PER_BLOCK:
        return False

    # Rule 3: Numbers kept in registers (TM * TN sums + TM + TN operands) <= 255
    if (TM * TN + TM + TN) > MAX_REGISTERS_PER_THREAD:
        return False

    # Rule 4: Two chunks in shared memory must fit in 48 KB
    if (BM + BN) * BK * 4 > MAX_SHARED_BYTES_PER_BLOCK:
        return False

    return True


def make_kernel(BM, BN, BK, WM, WN, TM, TN):
    assert is_valid_config(BM, BN, BK, WM, WN, TM, TN), "Invalid configuration!"
    NT = (BM // TM) * (BN // TN)

    @cuda.jit
    def matmul_tuned(matA, matB, matC, m, n, k):
        tileA = cuda.shared.array(shape=(BK, BM), dtype=cp.float32)
        tileB = cuda.shared.array(shape=(BK, BN), dtype=cp.float32)

        tid = cuda.threadIdx.x
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_row = (warp_id // (BN // WN)) * WM
        warp_col = (warp_id % (BN // WN)) * WN

        thread_row_base = warp_row + (lane_id // (WN // TN)) * TM
        thread_col_base = warp_col + (lane_id % (WN // TN)) * TN

        block_row = cuda.blockIdx.y * BM
        block_col = cuda.blockIdx.x * BN

        thread_results = cuda.local.array((TM, TN), dtype=cp.float32)
        regM = cuda.local.array(TM, dtype=cp.float32)
        regN = cuda.local.array(TN, dtype=cp.float32)

        for i in range(TM):
            for j in range(TN):
                thread_results[i, j] = float32(0.0)

        for bk_offset in range(0, k, BK):
            # Phase 1a: Load 4 neighbouring floats of A and store transposed in tileA
            for i in range(tid, (BM * BK) // 4, NT):
                r = i // (BK // 4)
                c = (i % (BK // 4)) * 4
                tileA[c + 0, r] = matA[block_row + r, bk_offset + c + 0]
                tileA[c + 1, r] = matA[block_row + r, bk_offset + c + 1]
                tileA[c + 2, r] = matA[block_row + r, bk_offset + c + 2]
                tileA[c + 3, r] = matA[block_row + r, bk_offset + c + 3]

            # Phase 1b: Load 4 neighbouring floats of B into tileB
            for i in range(tid, (BK * BN) // 4, NT):
                r = i // (BN // 4)
                c = (i % (BN // 4)) * 4
                tileB[r, c + 0] = matB[bk_offset + r, block_col + c + 0]
                tileB[r, c + 1] = matB[bk_offset + r, block_col + c + 1]
                tileB[r, c + 2] = matB[bk_offset + r, block_col + c + 2]
                tileB[r, c + 3] = matB[bk_offset + r, block_col + c + 3]

            cuda.syncthreads()

            # Phase 2: Read 4 floats at a time from tileA and tileB, then outer product
            for dot_k in range(BK):
                for i in range(0, TM, 4):
                    regM[i + 0] = tileA[dot_k, thread_row_base + i + 0]
                    regM[i + 1] = tileA[dot_k, thread_row_base + i + 1]
                    regM[i + 2] = tileA[dot_k, thread_row_base + i + 2]
                    regM[i + 3] = tileA[dot_k, thread_row_base + i + 3]

                for j in range(0, TN, 4):
                    regN[j + 0] = tileB[dot_k, thread_col_base + j + 0]
                    regN[j + 1] = tileB[dot_k, thread_col_base + j + 1]
                    regN[j + 2] = tileB[dot_k, thread_col_base + j + 2]
                    regN[j + 3] = tileB[dot_k, thread_col_base + j + 3]

                for i in range(TM):
                    for j in range(TN):
                        thread_results[i, j] += regM[i] * regN[j]

            cuda.syncthreads()

        # Write TM x TN patch results from registers back to global memory matC
        for i in range(TM):
            for j in range(TN):
                matC[block_row + thread_row_base + i, block_col + thread_col_base + j] = thread_results[i, j]

    return matmul_tuned, NT


CANDIDATE_CONFIGS = [
    (64,   64,  8, 16, 32, 4, 4),   # 64x64 tile, BK=8,  16x32 warp tile, 4x4 patch (256 threads)
    (64,   64, 16, 16, 32, 4, 4),   # 64x64 tile, BK=16, 16x32 warp tile, 4x4 patch (256 threads)
    (64,   64, 32, 16, 32, 4, 4),   # 64x64 tile, BK=32, 16x32 warp tile, 4x4 patch (256 threads)
    (64,   64, 16, 32, 32, 8, 4),   # 64x64 tile, BK=16, 32x32 warp tile, 8x4 patch (128 threads)
    (64,   64, 32, 32, 32, 8, 4),   # 64x64 tile, BK=32, 32x32 warp tile, 8x4 patch (128 threads)
    (64,  128, 16, 16, 32, 4, 4),   # 64x128 tile, BK=16, 16x32 warp tile, 4x4 patch (512 threads)
    (64,  128, 32, 16, 32, 4, 4),   # Chapter 5 fastest Kernel 6 shape at n=1024 (64x128, BK=32, 4x4)
    (64,  128, 16, 32, 32, 8, 4),   # 64x128 tile, BK=16, 32x32 warp tile, 8x4 patch (256 threads)
    (128, 128,  8, 32, 64, 8, 8),   # Boehm's 128x128 tile, BK=8,  8x8 patch (256 threads)
    (128, 128, 16, 32, 64, 8, 8),   # 128x128 tile, BK=16, 8x8 patch (256 threads)
    (128, 128, 32, 32, 64, 8, 8),   # Chapter 5 fastest Kernel 6 shape at n=4096 (128x128, BK=32, 8x8)
]


def benchmark_config(config, matA, matB, matC, m, n, k, warmup=3, repeats=10):
    BM, BN, BK, WM, WN, TM, TN = config
    kernel_fn, NT = make_kernel(BM, BN, BK, WM, WN, TM, TN)
    threads_per_block = (NT, 1)
    blocks_per_grid = (ceil_div(n, BN), ceil_div(m, BM))

    # perform warmup steps
    for _ in range(warmup):
        kernel_fn[blocks_per_grid, threads_per_block](matA, matB, matC, m, n, k)
    cuda.synchronize()

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # perform actual runs
    start_event.record()
    for _ in range(repeats):
        kernel_fn[blocks_per_grid, threads_per_block](matA, matB, matC, m, n, k)
    end_event.record()
    end_event.synchronize()

    avg_ms = cp.cuda.get_elapsed_time(start_event, end_event) / repeats
    avg_us = avg_ms * 1000.0
    tflops = (2.0 * m * n * k / (avg_ms * 1e-3)) / 1e12
    return avg_us, tflops, kernel_fn, blocks_per_grid, threads_per_block

m = 1024
k = 1024
n = 1024
matA = cp.ones(shape=(m, k), dtype=cp.float32) * 2.0
matB = cp.ones(shape=(k, n), dtype=cp.float32) * 3.0
matC = cp.zeros(shape=(m, n), dtype=cp.float32)

valid_configs = [cfg for cfg in CANDIDATE_CONFIGS if is_valid_config(*cfg)]
print(f"Autotuning Kernel 7 across {len(valid_configs)} configurations on ({m}x{k}) @ ({k}x{n})...")
print("-" * 86)
print(f"{'Config (BM, BN, BK, WM, WN, TM, TN)':<38} | {'Threads':>7} | {'Blocks':>6} | {'Time (us)':>10} | {'TFLOP/s':>8}")
print("-" * 86)
results = []
for cfg in valid_configs:
    BM, BN, BK, WM, WN, TM, TN = cfg
    nt = (BM // TM) * (BN // TN)
    num_blocks = ceil_div(n, BN) * ceil_div(m, BM)
    avg_us, tflops, kernel_fn, grid, block = benchmark_config(cfg, matA, matB, matC, m, n, k)
    results.append((avg_us, tflops, cfg, kernel_fn, grid, block))
    print(f"{str(cfg):<38} | {nt:>7} | {num_blocks:>6} | {avg_us:>10.1f} | {tflops:>8.2f}")
results.sort(key=lambda x: x[0])
best_us, best_tflops, best_cfg, best_kernel, best_grid, best_block = results[0]
print("-" * 86)
print(f"Fastest Kernel 7 Config : {best_cfg} -> {best_us:.1f} us ({best_tflops:.2f} TFLOP/s)")
print("-" * 86)

# Run fastest kernel and verify against NumPy
matC.fill(0.0)
best_kernel[best_grid, best_block](matA, matB, matC, m, n, k)
cuda.synchronize()

matA_np = cp.asnumpy(matA)
matB_np = cp.asnumpy(matB)
matC_np = cp.asnumpy(matC)

np.testing.assert_allclose(matC_np, matA_np @ matB_np, rtol=1e-5, atol=1e-5)
