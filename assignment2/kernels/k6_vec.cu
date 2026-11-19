// Kernel 6: kernel 5 with 16-byte moves (Lecture 9): float4 loads from global memory, A's chunk stored in shared
// memory turned on its side so that a thread's TM values of A sit side by side, and float4 reads from shared memory.
// Sizes: -DBM -DBN -DBK -DTM -DTN; a2.py only sends sizes where TM, TN, BK, BM and BN are multiples of 4.
// Launch (fixed by a2.py): grid (N/BN, M/BM), block ((BM/TM) * (BN/TN)).
// Before you write it: Problem 4.
#define NT ((BM / TM) * (BN / TN))
extern "C" __global__ void __launch_bounds__(NT) sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
}
