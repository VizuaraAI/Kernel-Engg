// Kernel 5: 2D tiling (Lecture 9). Each thread owns a TM x TN patch of the BM x BN tile and builds it from outer
// products: TM values of A's chunk and TN values of B's chunk into registers at each step k.
// Sizes: -DBM -DBN -DBK -DTM -DTN. Launch (fixed by a2.py): grid (N/BN, M/BM), block ((BM/TM) * (BN/TN)).
// Before you write it: Problem 3.
#define NT ((BM / TM) * (BN / TN))
extern "C" __global__ void __launch_bounds__(NT) sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
}
