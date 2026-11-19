// Kernel 7: warp tiling on top of kernel 6 (Lecture 9, Boehm's kernel 10). The BM x BN tile is split into WM x WN
// warp tiles, one for each warp. A warp tile is covered in WMITER x WNITER steps by sub-tiles of WSUBM x WSUBN; inside
// a sub-tile each of the warp's 32 threads owns a TM x TN patch, so each thread owns WMITER x WNITER patches.
// Sizes: -DBM -DBN -DBK -DWM -DWN -DWNITER -DTM -DTN; a2.py only sends sizes where every sub-tile is exactly 32
// patches and TN, BK, BM and BN are multiples of 4 (TM need not be).
// Launch (fixed by a2.py): grid (N/BN, M/BM), block (32 * (BM/WM) * (BN/WN)).
// Before you write it: Problem 5.
#define NT (32 * (BM / WM) * (BN / WN))
#define WMITER ((WM * WN) / (32 * TM * TN * WNITER))
#define WSUBM (WM / WMITER)
#define WSUBN (WN / WNITER)
extern "C" __global__ void __launch_bounds__(NT) sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
}
