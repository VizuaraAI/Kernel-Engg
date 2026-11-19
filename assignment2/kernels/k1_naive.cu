// Kernel 1: one thread for each element of C. threadIdx.x counts DOWN a column of C.
// Launch: grid (M/32, N/32), block (32, 32).   A is M x K, B is K x N, C is M x N, all row after row.
extern "C" __global__ void sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * 32 + threadIdx.x;      // neighbouring threads: neighbouring ROWS of C
    int col = blockIdx.y * 32 + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
