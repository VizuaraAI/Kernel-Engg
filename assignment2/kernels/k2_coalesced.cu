// Kernel 2: kernel 1 with neighbouring threads on neighbouring COLUMNS of C (Lecture 7, "two lines swapped").
// Launch (fixed by a2.py): grid (N/32, M/32), block (32, 32).
// A is M x K, B is K x N, C is M x N, all stored row after row. Write C = A x B; nothing else.
// Before you write it: Problem 1.
extern "C" __global__ void sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
}
