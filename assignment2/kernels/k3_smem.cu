// Kernel 3: 32 x 32 chunks of A and B in shared memory; one element of C for each thread (Lecture 7).
// Launch (fixed by a2.py): grid (N/32, M/32), block (32, 32). M, N and K are multiples of 32.
// Before you write it: Problem 1(c).
extern "C" __global__ void __launch_bounds__(1024) sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
}
