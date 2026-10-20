"""Assignment 1 practical: rooflines, arithmetic intensity and the memory hierarchy, measured on one H100.

    pip install -r requirements.txt
    modal setup                      # once: log in in the browser; Modal keeps your key on this machine
    modal run a1.py::ceilings        # Problem 7   copy speed against buffer size; the matmul and FP32 peaks
    modal run a1.py::roofline        # Problem 8   Y = X @ W for 1 to 8,192 tokens
    modal run a1.py::memory          # Problem 9   strided reads; bytes in flight per thread
    modal run a1.py::rope            # Problem 10  RoPE four ways, from 1 token to 131,072
    python plot.py                   # draws your figures from results/*.json, on your laptop

Each run rents one H100, prints what it measures and stops the GPU when it ends. A run takes under a minute of GPU
time, and an H100 costs about $3.95 an hour on Modal, billed by the second. The first run also builds the software
image, which takes a few minutes. Your laptop saves every number in results/<name>.json, together with the GPU's name,
its clocks, the time and the run's IDs (the app ID, ap-..., is the one modal.com/apps shows).

You do not write a kernel in this assignment. The settings you are asked to change are marked TUNE. The CUDA kernels
are in strings below, compiled on the GPU machine with NVRTC and launched from Python with CuPy, as in Lecture 3's
dog.py; the PyTorch parts use torch 2.14.

Every time comes from CUDA events: a few warm-up launches, then the median of many timed launches. Unless a table
says otherwise, the 50 MB L2 is flushed before each timed launch (we write 256 MB of zeros), so that the data has to
come from HBM. Your numbers will be close to ours, not identical, and they move a little from run to run.
"""
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
GPU = "H100"    # TUNE (bonus B1 only): another Modal GPU, for example "L4" or "A100-80GB"

app = modal.App("a1-rooflines")
# The base image is pinned by digest and every Python package by version and hash (image-requirements.txt), so the GPU
# machine runs exactly the software we tested: torch 2.14.1, CuPy 14.2.0 and CUDA 13.0.
image = (modal.Image.from_registry(
             "nvidia/cuda:13.0.0-devel-ubuntu24.04"
             "@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d", add_python="3.12")
         .entrypoint([])
         .pip_install_from_requirements(str(HERE / "image-requirements.txt")))

with image.imports():           # these exist only on the GPU machine; your laptop needs neither
    import cupy as cp
    import torch


# ============================================================================================== helpers (on the GPU)
def gpu_facts():
    """What the GPU says about itself, and when and where this run happened."""
    import subprocess

    p = torch.cuda.get_device_properties(0)
    q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,clocks.max.mem,power.limit",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    driver, sm_mhz, mem_mhz, watts = [x.strip() for x in q.split(",")]
    return {"gpu": p.name, "sms": p.multi_processor_count, "l2_mib": p.L2_cache_size / 2**20,
            "hbm_gib": round(p.total_memory / 2**30, 1), "driver": driver, "max_sm_clock_mhz": sm_mhz,
            "max_mem_clock_mhz": mem_mhz, "power_limit_w": watts, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "run_id": modal.current_function_call_id(),
            "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}


_SCRUB = []


def flush_l2():
    """Write 256 MB of zeros, so that nothing the next launch needs is still in the 50 MB L2."""
    if not _SCRUB:
        _SCRUB.append(torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda"))
    _SCRUB[0].zero_()


def time_us(fn, reps=30, warmup=3, flush=True):
    """Median time of fn on the GPU in microseconds, from CUDA events around each launch."""
    import statistics

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        if flush:
            flush_l2()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def burst_us(fn, n, trials=5):
    """GPU time per launch when n launches run back to back between two events (no flush in between)."""
    import statistics

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / n)
    return statistics.median(times)


def loop_us(fn, n=300):
    """Wall-clock time per call when the calls run back to back: what a program sees, launch costs included."""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6


def compile_kernel(source, name, defines=None):
    """Compile one CUDA C kernel with NVRTC (through CuPy) and return it, ready to launch."""
    head = "".join(f"#define {k} {v}\n" for k, v in (defines or {}).items())
    return cp.RawModule(code=head + source, options=("-std=c++17",)).get_function(name)


def as_cupy(t):
    """The same GPU memory as a torch tensor, seen by CuPy (no copy)."""
    return cp.from_dlpack(t)


# ================================================================================ Problem 7: the GPU's two ceilings
FMA_PEAK = r'''
extern "C" __global__ void fma_peak(float* out, int iters) {
    // Eight independent chains of multiply-adds, so that a lane never waits for its own previous result.
    float a = threadIdx.x * 1e-3f, b = a + 1.f, c = a + 2.f, d = a + 3.f;
    float e = a + 4.f, f = a + 5.f, g = a + 6.f, h = a + 7.f;
    #pragma unroll 16
    for (int i = 0; i < iters; ++i) {
        a = fmaf(a, 0.9999f, 0.0001f); b = fmaf(b, 0.9999f, 0.0001f);
        c = fmaf(c, 0.9999f, 0.0001f); d = fmaf(d, 0.9999f, 0.0001f);
        e = fmaf(e, 0.9999f, 0.0001f); f = fmaf(f, 0.9999f, 0.0001f);
        g = fmaf(g, 0.9999f, 0.0001f); h = fmaf(h, 0.9999f, 0.0001f);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a + b + c + d + e + f + g + h;   // so the work cannot be deleted
}
'''

COPY_SIZES = [m * 2**20 for m in (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096)]
#            TUNE (Problem 7a): bytes in each buffer, 1 MiB to 4 GiB
MATMUL_SIZES = [1024, 2048, 4096, 8192, 16384]  # square BF16 matmuls, N x N times N x N


@app.function(gpu=GPU, image=image, timeout=900)
def ceilings_gpu():
    facts = gpu_facts()
    print(f"{facts['gpu']} ({facts['sms']} SMs, L2 {facts['l2_mib']:.0f} MiB), run {facts['run_id']}")

    copies = []
    for n in COPY_SIZES:
        src = torch.randint(0, 255, (n,), dtype=torch.uint8, device="cuda")
        dst = torch.empty_like(src)
        # warm: many copies back to back, so src and dst stay in the L2 if they fit there.
        # cold: the L2 is flushed before each copy, so every byte comes from HBM and goes back to it.
        warm = burst_us(lambda: dst.copy_(src), n=max(4, min(200, 2**28 // n)))
        cold = time_us(lambda: dst.copy_(src), reps=50 if n < 2**28 else 15)
        copies.append({"bytes": n, "warm_us": warm, "cold_us": cold,
                       "warm_TBps": 2 * n / warm / 1e6, "cold_TBps": 2 * n / cold / 1e6})
        print(f"  copy {n / 2**20:8.0f} MiB   warm {2 * n / warm / 1e6:6.2f} TB/s   cold {2 * n / cold / 1e6:6.2f} TB/s")
        del src, dst

    matmuls = []
    for n in MATMUL_SIZES:
        a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        c = torch.empty(n, n, dtype=torch.bfloat16, device="cuda")
        t = time_us(lambda: torch.matmul(a, b, out=c), reps=20 if n < 16384 else 8)
        matmuls.append({"n": n, "us": t, "TFLOPs": 2 * n**3 / t / 1e6})
        print(f"  matmul {n:6d}^3 bf16   {2 * n**3 / t / 1e6:7.1f} TFLOP/s")
        del a, b, c


    k = compile_kernel(FMA_PEAK, "fma_peak")
    iters, threads, blocks = 1 << 14, 256, facts["sms"] * 8                    # 2,048 threads on every SM
    out = cp.empty(blocks * threads, dtype=cp.float32)
    t = time_us(lambda: k((blocks,), (threads,), (out, cp.int32(iters))), reps=20, flush=False)
    fp32 = {"us": t, "TFLOPs": 2 * 8 * iters * blocks * threads / t / 1e6}
    print(f"  fp32 multiply-adds on the CUDA cores   {fp32['TFLOPs']:6.1f} TFLOP/s")
    return {"facts": facts, "copy": copies, "matmul": matmuls, "fp32_fma": fp32}


@app.local_entrypoint()
def ceilings():
    r = ceilings_gpu.remote()
    save("ceilings", r)
    print(header(r["facts"]))
    print("copy: dst.copy_(src), counting the bytes read and the bytes written")
    print("  warm: copies back to back, so the L2 keeps both buffers if they fit; cold: L2 flushed before each copy")
    print(f"  {'buffer':>10}   {'warm TB/s':>9}   {'cold TB/s':>9}")
    for c in r["copy"]:
        print(f"  {fmt_bytes(c['bytes']):>10}   {c['warm_TBps']:9.2f}   {c['cold_TBps']:9.2f}")
    print("matmul: torch.matmul in bf16 (tensor cores)")
    for m in r["matmul"]:
        print(f"  {m['n']:>6} x {m['n']} x {m['n']}   {m['us']:10.1f} us   {m['TFLOPs']:7.1f} TFLOP/s")
    print(f"fp32 multiply-adds (CUDA cores): {r['fp32_fma']['TFLOPs']:.1f} TFLOP/s")


# ================================================================== Problem 8: trace the roofline with real matmuls
K_DIM, N_DIM = 4096, 12288                    # Qwen3-8B's up projection: 4,096 in, 12,288 out
TOKENS = [2**k for k in range(14)]            # 1, 2, 4, ..., 8,192 tokens in the batch (the M of the matmul)
FINE_TOKENS = list(range(256, 513, 16))       # TUNE (Problem 8c): a fine sweep of batch sizes


@app.function(gpu=GPU, image=image, timeout=900)
def roofline_gpu():
    facts = gpu_facts()
    print(f"{facts['gpu']}, run {facts['run_id']}")
    w = torch.randn(K_DIM, N_DIM, dtype=torch.bfloat16, device="cuda") * 0.02
    rows = []
    for m in sorted(set(TOKENS + FINE_TOKENS)):
        x = torch.randn(m, K_DIM, dtype=torch.bfloat16, device="cuda")
        y = torch.empty(m, N_DIM, dtype=torch.bfloat16, device="cuda")
        t = time_us(lambda: torch.matmul(x, w, out=y), reps=30)
        rows.append({"m": m, "us": t, "TFLOPs": 2 * m * K_DIM * N_DIM / t / 1e6})
        print(f"  M = {m:5d}   {t:9.2f} us   {2 * m * K_DIM * N_DIM / t / 1e6:7.1f} TFLOP/s")
    return {"facts": facts, "k": K_DIM, "n": N_DIM, "rows": rows, "fine": FINE_TOKENS}


@app.local_entrypoint()
def roofline():
    r = roofline_gpu.remote()
    save("roofline", r)
    print(header(r["facts"]))
    print(f"Y = X @ W with W of {r['k']} x {r['n']} in bf16, X of M x {r['k']}")
    print(f"  {'M':>5}   {'time (us)':>10}   {'TFLOP/s':>8}")
    for row in r["rows"]:
        print(f"  {row['m']:5d}   {row['us']:10.2f}   {row['TFLOPs']:8.1f}")


# =========================================================== Problem 9: crates, coalescing and bytes in flight
STRIDED_COPY = r'''
extern "C" __global__ void strided_copy(const float* __restrict__ src, float* __restrict__ dst,
                                        long long n, int stride) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // which float am I?
    if (i < n) dst[i] = src[i * stride];       // neighbouring threads read floats `stride` apart; writes stay side by side
}
'''

GRAY = r'''
// VEC pixels per load (4, 8 or 16 bytes) and ILP independent loads per thread, set by the #defines above.
#if VEC == 1
typedef unsigned int PIX;  typedef unsigned char GRAY;
#elif VEC == 2
typedef uint2 PIX;         typedef uchar2 GRAY;
#else
typedef uint4 PIX;         typedef uchar4 GRAY;
#endif

__device__ __forceinline__ unsigned char gray_of(unsigned int p) {          // the blog's formula, one RGBA pixel
    return (unsigned char)(0.299f * (p & 255) + 0.587f * ((p >> 8) & 255) + 0.114f * ((p >> 16) & 255));
}
#if VEC == 1
__device__ __forceinline__ GRAY to_gray(PIX p) { return gray_of(p); }
#elif VEC == 2
__device__ __forceinline__ GRAY to_gray(PIX p) { return make_uchar2(gray_of(p.x), gray_of(p.y)); }
#else
__device__ __forceinline__ GRAY to_gray(PIX p) {
    return make_uchar4(gray_of(p.x), gray_of(p.y), gray_of(p.z), gray_of(p.w));
}
#endif

extern "C" __global__ void gray(const PIX* __restrict__ rgb, GRAY* __restrict__ out) {
    long long first = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;      // the whole grid's threads, one load each
    PIX p[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; ++j) p[j] = rgb[first + j * step];    // all ILP loads leave before any comes back
    #pragma unroll
    for (int j = 0; j < ILP; ++j) out[first + j * step] = to_gray(p[j]);
}
'''

STRIDES = [1, 2, 4, 8, 16, 32]       # TUNE (Problem 9a): floats between the reads of neighbouring threads
STRIDED_N = 1 << 24                  # floats copied: 64 MiB read (useful bytes) and 64 MiB written
GRAY_VEC = [1, 2, 4]                 # TUNE (Problem 9b): pixels per load
GRAY_ILP = [1, 2, 4]                 # TUNE (Problem 9b): independent loads per thread
GRAY_BLOCK = [128, 256, 512, 1024]   # TUNE (Problem 9b): threads per block
DOGS = 64                            # 64 photos the size of Lecture 3's dog: 1024 x 768 RGBA pixels each


@app.function(gpu=GPU, image=image, timeout=900)
def memory_gpu():
    facts = gpu_facts()
    print(f"{facts['gpu']}, run {facts['run_id']}")

    k = compile_kernel(STRIDED_COPY, "strided_copy")
    src = torch.randn(STRIDED_N * max(STRIDES), dtype=torch.float32, device="cuda")
    dst = torch.empty(STRIDED_N, dtype=torch.float32, device="cuda")
    s_, d_ = as_cupy(src), as_cupy(dst)
    strided = []
    for s in STRIDES:
        launch = lambda: k(((STRIDED_N + 255) // 256,), (256,), (s_, d_, cp.int64(STRIDED_N), cp.int32(s)))
        launch()
        assert torch.equal(dst, src[::s][:STRIDED_N]), f"stride {s}: wrong result"
        t = time_us(launch)
        strided.append({"stride": s, "us": t, "useful_TBps": 8 * STRIDED_N / t / 1e6})
        print(f"  stride {s:3d}   {t:8.1f} us   useful {8 * STRIDED_N / t / 1e6:5.2f} TB/s")
    del src, dst

    n_px = DOGS * 1024 * 768
    rgb = torch.randint(0, 2**31 - 1, (n_px,), dtype=torch.int32, device="cuda")    # random RGBA pixels
    ref = None
    gray_rows = []
    for vec in GRAY_VEC:
        for ilp in GRAY_ILP:
            kern = compile_kernel(GRAY, "gray", {"VEC": vec, "ILP": ilp})
            for block in GRAY_BLOCK:
                per_block = vec * ilp * block
                assert n_px % per_block == 0
                out = torch.empty(n_px, dtype=torch.uint8, device="cuda")
                r_, o_ = as_cupy(rgb), as_cupy(out)
                launch = lambda: kern((n_px // per_block,), (block,), (r_, o_))
                launch()
                if ref is None:
                    ref = out.clone()
                assert torch.equal(out, ref), f"VEC {vec} ILP {ilp} block {block}: wrong result"
                t = time_us(launch)
                tbps = 5 * n_px / t / 1e6                         # 4 bytes read and 1 written for every pixel
                gray_rows.append({"vec": vec, "ilp": ilp, "block": block, "bytes_per_thread": 4 * vec * ilp,
                                  "us": t, "TBps": tbps})
                print(f"  gray  VEC {vec}  ILP {ilp}  block {block:4d}   {t:7.1f} us   {tbps:5.2f} TB/s")
    return {"facts": facts, "strided_n": STRIDED_N, "strided": strided, "gray_pixels": n_px, "gray": gray_rows}


@app.local_entrypoint()
def memory():
    r = memory_gpu.remote()
    save("memory", r)
    print(header(r["facts"]))
    print(f"strided copy of {r['strided_n']:,} floats: dst[i] = src[i * stride]")
    print(f"  {'stride':>6}   {'time (us)':>9}   {'useful TB/s':>11}")
    for row in r["strided"]:
        print(f"  {row['stride']:6d}   {row['us']:9.1f}   {row['useful_TBps']:11.2f}")
    print(f"grayscale of {r['gray_pixels']:,} pixels (64 dogs): 4 bytes in, 1 byte out per pixel")
    print(f"  {'VEC':>3} {'ILP':>3} {'block':>5}  {'bytes/thread':>12}   {'time (us)':>9}   {'TB/s':>5}")
    for row in r["gray"]:
        print(f"  {row['vec']:3d} {row['ilp']:3d} {row['block']:5d}  {row['bytes_per_thread']:12d}"
              f"   {row['us']:9.1f}   {row['TBps']:5.2f}")


# ============================================================================== Problem 10: RoPE in the hierarchy
ROPE = r'''
// Rotary embedding, in place, on q [S, HQ, 128] and k [S, HK, 128] in bf16, in one launch (as serving engines do).
// For each head vector x and pair i < 64:  x[i] <- x[i] cos - x[i+64] sin,  x[i+64] <- x[i+64] cos + x[i] sin,
// with cos and sin read from fp32 tables of shape [S, 64]. PAIRS and HEAD_MAJOR are set by the #defines above.
#if PAIRS == 1
typedef unsigned short VT;    // 2 bytes: 1 bf16
#elif PAIRS == 2
typedef unsigned int VT;      // 4 bytes: 2 bf16
#elif PAIRS == 4
typedef uint2 VT;             // 8 bytes: 4 bf16
#else
typedef uint4 VT;             // 16 bytes: 8 bf16
#endif

__device__ __forceinline__ float bf16_to_float(unsigned short u) { return __uint_as_float(((unsigned int)u) << 16); }
__device__ __forceinline__ unsigned short float_to_bf16(float f) {    // round to nearest even
    unsigned int x = __float_as_uint(f);
    return (unsigned short)((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

extern "C" __global__ void rope(unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                                const float* __restrict__ cos_t, const float* __restrict__ sin_t,
                                int S, int HQ, int HK) {
    const int per_vec = 64 / PAIRS;                        // threads that share one head vector of 128 numbers
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long v = t / per_vec;                             // which head vector this thread works on
    int part = (int)(t % per_vec);
    int H = HQ + HK;
    if (v >= (long long)S * H) return;
#if HEAD_MAJOR
    int h = (int)(v / S), s = (int)(v % S);                // neighbouring threads walk the tokens of one head
#else
    int s = (int)(v / H), h = (int)(v % H);                // neighbouring threads walk the heads of one token
#endif
    unsigned short* x = h < HQ ? q + ((long long)s * HQ + h) * 128 : k + ((long long)s * HK + (h - HQ)) * 128;
    int i0 = part * PAIRS;
    union { VT v; unsigned short e[PAIRS]; } a, b;
    a.v = *(const VT*)(x + i0);                            // first half:  x[i0 .. i0 + PAIRS)
    b.v = *(const VT*)(x + 64 + i0);                       // second half: x[64 + i0 .. 64 + i0 + PAIRS)
    #pragma unroll
    for (int j = 0; j < PAIRS; ++j) {
        float c = cos_t[(long long)s * 64 + i0 + j], sn = sin_t[(long long)s * 64 + i0 + j];
        float x1 = bf16_to_float(a.e[j]), x2 = bf16_to_float(b.e[j]);
        a.e[j] = float_to_bf16(x1 * c - x2 * sn);
        b.e[j] = float_to_bf16(x2 * c + x1 * sn);
    }
    *(VT*)(x + i0) = a.v;
    *(VT*)(x + 64 + i0) = b.v;
}
'''

EMPTY = r'extern "C" __global__ void empty() {}'

SEQ = [1, 16, 128, 1024, 4096, 16384, 65536, 131072]   # TUNE (Problem 10b): tokens S
HQ, HK, HEAD_DIM = 32, 8, 128                           # Qwen3-8B: 32 query heads and 8 key heads of 128 numbers
THETA = 1_000_000.0                                     # Qwen3's RoPE base
CUDA_PAIRS = 2                                          # TUNE (Problem 10c): rotation pairs per thread: 1, 2, 4, 8
ORDER_SEQ = [4096, 131072]                              # TUNE (Problem 10c): sizes for the order test
ORDER_PAIRS = [1, 2, 4, 8]


def rotate_half(x):
    x1, x2 = x[..., :64], x[..., 64:]
    return torch.cat((-x2, x1), dim=-1)


def rope_eager(q, k, cos, sin):
    """RoPE as Hugging Face's Llama and Qwen3 code writes it; cos and sin are [S, 1, 128] in bf16."""
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


@app.function(gpu=GPU, image=image, timeout=1200)
def rope_gpu():
    facts = gpu_facts()
    print(f"{facts['gpu']}, run {facts['run_id']}")
    try:                                   # one compiled kernel for every S, as a fair comparison
        torch._dynamo.config.recompile_limit = 32
    except AttributeError:
        torch._dynamo.config.cache_size_limit = 32
    rope_compiled = torch.compile(rope_eager, dynamic=False)

    kernels = {(p, hm): compile_kernel(ROPE, "rope", {"PAIRS": p, "HEAD_MAJOR": hm})
               for p in sorted(set(ORDER_PAIRS + [CUDA_PAIRS])) for hm in (0, 1)}
    empty = compile_kernel(EMPTY, "empty")

    def tables(S):
        pos = torch.arange(S, dtype=torch.float64, device="cuda")
        inv = THETA ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device="cuda") / HEAD_DIM)
        ang = torch.outer(pos, inv)                                           # [S, 64]
        cos32, sin32 = ang.cos().float().contiguous(), ang.sin().float().contiguous()
        cos16 = torch.cat((cos32, cos32), -1).to(torch.bfloat16)[:, None, :]  # [S, 1, 128], as in Hugging Face
        sin16 = torch.cat((sin32, sin32), -1).to(torch.bfloat16)[:, None, :]
        return cos32, sin32, cos16, sin16

    def launcher(q, k, cos32, sin32, S, pairs, head_major):
        kern = kernels[(pairs, head_major)]
        threads = S * (HQ + HK) * (64 // pairs)
        # CuPy has no bf16 type, so it sees q and k as raw 16-bit integers; the kernel only needs the bits
        args = (as_cupy(q.view(torch.int16)), as_cupy(k.view(torch.int16)), as_cupy(cos32), as_cupy(sin32),
                cp.int32(S), cp.int32(HQ), cp.int32(HK))
        return lambda: kern(((threads + 255) // 256,), (256,), args)

    gpu_rows, loop_rows, order_rows, errors = [], [], [], {}
    for S in SEQ:
        q = torch.randn(S, HQ, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(S, HK, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        cos32, sin32, cos16, sin16 = tables(S)
        both = torch.empty(S * (HQ + HK) * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        both2 = torch.empty_like(both)
        qc, kc = q.clone(), k.clone()                         # the CUDA kernel works in place, on its own copies
        cuda_run = launcher(qc, kc, cos32, sin32, S, CUDA_PAIRS, 0)

        if S == 4096:                                         # correctness, against an fp32 reference
            ref_q = q.float() * torch.cat((cos32, cos32), -1)[:, None] + rotate_half(q.float()) * \
                torch.cat((sin32, sin32), -1)[:, None]
            ref_k = k.float() * torch.cat((cos32, cos32), -1)[:, None] + rotate_half(k.float()) * \
                torch.cat((sin32, sin32), -1)[:, None]
            eq, ek = rope_eager(q, k, cos16, sin16)
            cq, ck = rope_compiled(q, k, cos16, sin16)
            cuda_run()
            for name, (a, b) in {"eager": (eq, ek), "compiled": (cq, ck), "cuda": (qc, kc)}.items():
                errors[name] = max((a.float() - ref_q).abs().max().item(), (b.float() - ref_k).abs().max().item())
            print("  max |error| against fp32 at S = 4096: " + ", ".join(f"{n} {e:.4f}" for n, e in errors.items()))
            qc.copy_(q)
            kc.copy_(k)

        row = {"S": S,
               "copy_us": time_us(lambda: both2.copy_(both)),
               "eager_us": time_us(lambda: rope_eager(q, k, cos16, sin16)),
               "compiled_us": time_us(lambda: rope_compiled(q, k, cos16, sin16)),
               "cuda_us": time_us(cuda_run)}
        gpu_rows.append(row)
        print(f"  S = {S:6d}   copy {row['copy_us']:8.1f}   eager {row['eager_us']:8.1f}   "
              f"compiled {row['compiled_us']:8.1f}   cuda {row['cuda_us']:8.1f}   (us)")

        if S <= 128:
            lr = {"S": S,
                  "eager_us": loop_us(lambda: rope_eager(q, k, cos16, sin16)),
                  "compiled_us": loop_us(lambda: rope_compiled(q, k, cos16, sin16)),
                  "cuda_us": loop_us(cuda_run),
                  "empty_kernel_us": loop_us(lambda: empty((1,), (32,), ()))}
            loop_rows.append(lr)
            print(f"           back to back: eager {lr['eager_us']:6.1f}   compiled {lr['compiled_us']:6.1f}   "
                  f"cuda {lr['cuda_us']:6.1f}   empty kernel {lr['empty_kernel_us']:6.1f}   (us per call)")

        if S in ORDER_SEQ:
            for pairs in ORDER_PAIRS:
                for hm in (0, 1):
                    t = time_us(launcher(qc, kc, cos32, sin32, S, pairs, hm))
                    order_rows.append({"S": S, "pairs": pairs, "order": "head-major" if hm else "token-major",
                                       "us": t})
                    print(f"           cuda  pairs {pairs}  {'head-major ' if hm else 'token-major'}   {t:8.1f} us")
        del q, k, qc, kc, both, both2, cos32, sin32, cos16, sin16
        torch.cuda.empty_cache()
    return {"facts": facts, "hq": HQ, "hk": HK, "head_dim": HEAD_DIM, "cuda_pairs": CUDA_PAIRS,
            "gpu": gpu_rows, "loop": loop_rows, "order": order_rows, "errors_at_4096": errors}


@app.local_entrypoint()
def rope():
    r = rope_gpu.remote()
    save("rope", r)
    print(header(r["facts"]))
    print(f"RoPE on q [S, {r['hq']}, 128] and k [S, {r['hk']}, 128] in bf16. GPU time per call, L2 flushed (us):")
    print(f"  {'S':>6}   {'copy':>8}   {'eager':>8}   {'compiled':>8}   {'cuda':>8}")
    for row in r["gpu"]:
        print(f"  {row['S']:6d}   {row['copy_us']:8.1f}   {row['eager_us']:8.1f}   {row['compiled_us']:8.1f}"
              f"   {row['cuda_us']:8.1f}")
    print("Back to back, wall clock per call, launch costs included (us):")
    print(f"  {'S':>6}   {'eager':>8}   {'compiled':>8}   {'cuda':>8}   {'empty kernel':>12}")
    for row in r["loop"]:
        print(f"  {row['S']:6d}   {row['eager_us']:8.1f}   {row['compiled_us']:8.1f}   {row['cuda_us']:8.1f}"
              f"   {row['empty_kernel_us']:12.1f}")
    print("The CUDA kernel, two orders of work (us):")
    for row in r["order"]:
        print(f"  S = {row['S']:6d}   pairs {row['pairs']}   {row['order']:<11}   {row['us']:8.1f}")
    print("max |error| against an fp32 reference at S = 4096: "
          + ", ".join(f"{n} {e:.4f}" for n, e in r["errors_at_4096"].items()))


# ======================================================================================== helpers (on your laptop)
def save(name, result):
    result["facts"]["app_id"] = app.app_id          # the ap-... ID that modal.com/apps shows for this run
    if GPU != "H100":                               # bonus B1: keep the H100 results; save this GPU's next to them
        name = f"{name}_{GPU.replace('-', '_')}"
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"\nsaved {path}")


def header(f):
    return (f"\n{f['gpu']} ({f['sms']} SMs, L2 {f['l2_mib']:.0f} MiB, max SM clock {f['max_sm_clock_mhz']} MHz, "
            f"driver {f['driver']})\napp {f['app_id']}, run {f['run_id']}, {f['utc']}\n")


def fmt_bytes(n):
    return f"{n / 2**30:.0f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"
