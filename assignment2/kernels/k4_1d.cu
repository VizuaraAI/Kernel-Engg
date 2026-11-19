// Kernel 4: 1D tiling (Lecture 7). A block computes a BM x BN tile of C with BM * BN / TM threads; each thread owns TM
// elements of one column of the tile. The sizes arrive as -D flags: -DBM=64 -DBN=64 -DBK=8 -DTM=8, for example.
// Launch (fixed by a2.py): grid (N/BN, M/BM), block (BM * BN / TM). M % BM == N % BN == K % BK == 0.
// Before you write it: Problem 2.
#define NT (BM * BN / TM)            // threads in a block
extern "C" __global__ void __launch_bounds__(NT) sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
}
