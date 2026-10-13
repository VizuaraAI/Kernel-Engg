"""Lecture 3 practical: the dog from "The CUDA Programming Model" (vizuaraai.github.io/cuda-programming-model), on a
real NVIDIA H100.

    pip install modal
    modal setup                  # log in once in the browser; Modal keeps your key on this machine
    modal run dog.py             # a few minutes the first time (the image is built once), then under a minute

Every kernel below is CUDA C, written as close to the blog's code as we could, compiled on the GPU machine with NVRTC
and launched from Python with CuPy. One H100 runs them on the blog's own 1024 x 768 dog and measures the numbers on the
lecture's two "Measured" slides, "What the dog costs" and "Divergence":

    section 4    kernel A, rgb_to_gray: one thread per pixel, 16 x 16 blocks (pixel (579, 329) must come out as 36)
    section 10   the checkerboard branch and portrait mode, a count of the warps whose 32 threads disagree, and the same
                 branches made heavy (512 steps of arithmetic a side), so that running both sides shows in the time

An empty kernel shows the floor: no launch can take less. Each time comes from CUDA events, the median of many launches
after a warm-up. The images land in out/, checked pixel by pixel against the blog's own images in dog/blog/; the numbers
land in data/dog_kernels.json. The GPU costs about $3.95 an hour on Modal, so one run costs a few cents. The container
stops itself after five minutes whatever happens.
"""
import io
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("l3-dog")
image = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .pip_install("cupy-cuda13x==14.2.0", "numpy==2.3.5", "pillow==10.4.0"))

W, H = 1024, 768                    # the dog

# ===================================================================================================== the kernels
# Section 4. The blog's kernel; only extern "C" is added, so that CuPy can find it by name.
RGB_TO_GRAY = r'''
extern "C" __global__ void rgb_to_gray(const uchar4* rgb, unsigned char* gray,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // which column am I?
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // which row am I?
    if (x >= width || y >= height) return;           // guard the edges

    uchar4 p = rgb[y * width + x];                   // read my pixel
    gray[y * width + x] =
        (unsigned char)(0.299f*p.x + 0.587f*p.y + 0.114f*p.z);
}
'''

# Section 10. The checkerboard: half of every warp brightens, the other half darkens.
CHECKERBOARD = r'''
__device__ uchar4 brighten(uchar4 p) {          // x 1.45, held at 255
    return make_uchar4(fminf(p.x * 1.45f, 255.f), fminf(p.y * 1.45f, 255.f), fminf(p.z * 1.45f, 255.f), p.w);
}
__device__ uchar4 darken(uchar4 p) {            // x 0.55
    return make_uchar4(p.x * 0.55f, p.y * 0.55f, p.z * 0.55f, p.w);
}
extern "C" __global__ void checkerboard(const uchar4* rgb, uchar4* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = y * width + x;
    uchar4 p = rgb[i];
    if ((x + y) % 2 == 0) out[i] = brighten(p);   // half of every warp...
    else                  out[i] = darken(p);     // ...the other half
}
'''

# Section 10. Portrait mode: a mask tells each thread whether its pixel is dog or background.
PORTRAIT = r'''
extern "C" __global__ void portrait(const uchar4* rgb, const uchar4* blurred, const unsigned char* isDog,
                                    uchar4* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = y * width + x;
    uchar4 p = rgb[i];
    if (isDog[i]) out[i] = p;              // keep the puppy sharp
    else          out[i] = blurred[i];     // soften the background
}
'''

# Section 10. Counting divergence on the GPU itself: every warp takes a vote on the branch condition (__ballot_sync
# returns one bit per thread), and a warp whose 32 bits are neither all 0 nor all 1 must run both sides.
#   which 0: the checkerboard's condition   which 1: portrait mode's isDog   which 2: the top half of the image (y < 384)
CENSUS = r'''
extern "C" __global__ void census(const unsigned char* isDog, int width, int height, int which, unsigned int* diverged) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool inside = x < width && y < height;
    bool c = which == 0 ? ((x + y) % 2 == 0) : which == 1 ? (inside && isDog[y * width + x] != 0) : (y < height / 2);
    unsigned int votes = __ballot_sync(0xffffffffu, c);
    int lane = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (lane == 0 && votes != 0u && votes != 0xffffffffu) atomicAdd(diverged, 1u);
}
'''

# Section 10, made heavy. The blog's two branches are a few instructions each, so the time a warp spends on both sides
# hides behind its memory wait. Here each side does 512 steps of arithmetic (different arithmetic on each side, so the
# compiler cannot merge them), and the price of running both sides shows up in the clock.
HEAVY_BRANCH = r'''
#define STEPS 512
__device__ float lighter(float v) {
    #pragma unroll 1
    for (int k = 0; k < STEPS; ++k) v = v * 0.999f + 0.255f;     // drifts up toward 255
    return v;
}
__device__ float darker(float v) {
    #pragma unroll 1
    for (int k = 0; k < STEPS; ++k) v = v - v * 0.001f;          // decays toward 0
    return v;
}
extern "C" __global__ void heavy_branch(const unsigned char* gray, const unsigned char* isDog, unsigned char* out,
                                        int width, int height, int which) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = y * width + x;
    bool c = which == 0 ? ((x + y) % 2 == 0) : which == 1 ? (isDog[i] != 0) : (y < height / 2);
    float v = gray[i];
    if (c) v = lighter(v);
    else   v = darker(v);
    out[i] = (unsigned char)fminf(fmaxf(v, 0.f), 255.f);
}
'''

# Timing helpers. A spin kernel holds the GPU for a millisecond while Python queues the launches behind it, so the
# launches then run back to back and the CUDA events measure the GPU, not Python. An empty kernel shows the floor.
SPIN = r'''
extern "C" __global__ void spin(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
extern "C" __global__ void empty() {}
'''


# ===================================================================================================== on the GPU
@app.function(image=image, gpu="H100", max_containers=1, timeout=420)
def run(dog_png: bytes, mask_png: bytes, blog: dict) -> dict:
    import os
    import threading

    def watchdog(seconds=300):               # whatever happens, this container stops after five minutes
        time.sleep(seconds)
        print(f"watchdog: {seconds} s passed, stopping", flush=True)
        os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()

    import cupy as cp
    import numpy as np
    from PIL import Image, ImageFilter

    t_start = time.time()
    props = cp.cuda.runtime.getDeviceProperties(0)
    dev = dict(name=props["name"].decode(), sms=props["multiProcessorCount"], cc=f'{props["major"]}.{props["minor"]}',
               driver=cp.cuda.runtime.driverGetVersion(), runtime=cp.cuda.runtime.runtimeGetVersion(),
               nvrtc=".".join(map(str, cp.cuda.nvrtc.getVersion())), cupy=cp.__version__)
    print(f"{dev['name']}: {dev['sms']} SMs, compute capability {dev['cc']}, CUDA driver {dev['driver']}, "
          f"NVRTC {dev['nvrtc']}, CuPy {dev['cupy']}", flush=True)

    def kernel(src, name, *options):
        k = cp.RawKernel(src, name, options=tuple(options))
        k.compile()
        return k

    timing = kernel(SPIN, "spin")
    empty = kernel(SPIN, "empty")

    def gpu_us(launch, batch=20, trials=15):
        """Median microseconds per launch: CUDA events around `batch` launches queued behind a 1 ms spin, `trials`
        times, after one warm-up launch."""
        launch()
        cp.cuda.Device().synchronize()
        s, e, out = cp.cuda.Event(), cp.cuda.Event(), []
        for _ in range(trials):
            timing((1,), (1,), (np.uint64(1_000_000),))
            s.record()
            for _ in range(batch):
                launch()
            e.record()
            e.synchronize()
            out.append(cp.cuda.get_elapsed_time(s, e) * 1e3 / batch)
        return float(np.median(out))

    results = []

    def measure(key, name, launch, note="", batch=20, trials=15):
        t = round(gpu_us(launch, batch, trials), 3)
        results.append(dict(id=key, name=name, time_us=t, note=note))
        print(f"  measured {name}: {t:.2f} us", flush=True)
        return t

    # ---------------------------------------------------------------- the dog on the GPU
    rgb = np.asarray(Image.open(io.BytesIO(dog_png)).convert("RGB"))
    assert rgb.shape == (H, W, 3)
    rgba = np.dstack([rgb, np.full((H, W), 255, np.uint8)])
    mask = (np.asarray(Image.open(io.BytesIO(mask_png))) > 127).astype(np.uint8)
    d_rgba, d_mask = cp.asarray(rgba), cp.asarray(mask)
    grid16, block16 = ((W + 15) // 16, (H + 15) // 16), (16, 16)       # 64 x 48 blocks of 16 x 16 threads
    W_, H_ = np.int32(W), np.int32(H)
    floor_us = gpu_us(lambda: empty((1,), (32,), ()))
    print(f"an empty kernel takes {floor_us:.2f} us per launch measured this way", flush=True)
    pngs, checks = {}, {}

    def png(name, arr):
        b = io.BytesIO()
        Image.fromarray(arr).save(b, "PNG")
        pngs[name] = b.getvalue()

    def ref(name):
        return np.asarray(Image.open(io.BytesIO(blog[name])))

    def same(name, mine, theirs, what):
        n_same = int((mine == theirs).all(axis=-1).sum() if mine.ndim == 3 else (mine == theirs).sum())
        n_all = mine.shape[0] * mine.shape[1]
        diff = int(np.abs(mine.astype(int) - theirs.astype(int)).max())
        checks[name] = dict(against=what, identical_pixels=n_same, pixels=n_all, max_difference=diff)
        return n_same, diff

    # section 4: grayscale
    gray_k = kernel(RGB_TO_GRAY, "rgb_to_gray")
    d_gray = cp.empty((H, W), cp.uint8)
    measure("rgb_to_gray", "kernel A: grayscale, one dog", lambda: gray_k(grid16, block16, (d_rgba, d_gray, W_, H_)),
            note="reads 4 bytes and writes 1 per pixel; the dog is 3 MiB, so it stays in the 50 MB L2 cache between runs")
    gray = cp.asnumpy(d_gray)
    png("gray", gray)
    same("gray", gray, ref("gray.png"), "dog/blog/gray.png")
    showcase = dict(pixel=[579, 329], rgb=[int(v) for v in rgb[329, 579]], gray=int(gray[329, 579]))
    # the same kernel compiled with --fmad=false: every multiply and add rounded on its own, as NumPy does
    gray_nofma_k = kernel(RGB_TO_GRAY, "rgb_to_gray", "--fmad=false")
    d_gray2 = cp.empty((H, W), cp.uint8)
    gray_nofma_k(grid16, block16, (d_rgba, d_gray2, W_, H_))
    same("gray_without_fma", cp.asnumpy(d_gray2), ref("gray.png"), "dog/blog/gray.png, compiled with --fmad=false")

    # section 10: divergence
    check_k = kernel(CHECKERBOARD, "checkerboard")
    d_out = cp.empty_like(d_rgba)
    measure("checkerboard", "checkerboard", lambda: check_k(grid16, block16, (d_rgba, d_out, W_, H_)),
            note="branch bodies of a few instructions each")
    checker = cp.asnumpy(d_out)[..., :3]
    png("checker", checker)
    same("checker", checker, ref("checker.png"), "dog/blog/checker.png")
    blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(9)))      # the blog's own background
    d_blurred = cp.asarray(np.dstack([blurred, np.full((H, W), 255, np.uint8)]))
    port_k = kernel(PORTRAIT, "portrait")
    measure("portrait", "portrait mode",
            lambda: port_k(grid16, block16, (d_rgba, d_blurred, d_mask, d_out, W_, H_)),
            note="reads the pixel, the mask byte and (for background) the blurred pixel")
    portrait = cp.asnumpy(d_out)[..., :3]
    png("portrait", portrait)
    same("portrait", portrait, ref("portrait.png"), "dog/blog/portrait.png")
    census_k = kernel(CENSUS, "census")
    divergence = {}
    for which, label in ((0, "checkerboard"), (1, "portrait mode"), (2, "top half (y < 384)")):
        d_count = cp.zeros(1, cp.uint32)
        census_k(grid16, block16, (d_mask, W_, H_, np.int32(which), d_count))
        divergence[label] = int(d_count.get()[0])
    heavy_k = kernel(HEAVY_BRANCH, "heavy_branch")
    d_heavy = cp.empty((H, W), cp.uint8)
    heavy = {}
    for which, label in ((0, "checkerboard"), (1, "portrait mode"), (2, "top half (y < 384)")):
        heavy[label] = measure(
            f"heavy_{which}", f"heavy branch: {label}",
            lambda which=which: heavy_k(grid16, block16, (d_gray, d_mask, d_heavy, W_, H_, np.int32(which))),
            note="each side of the branch runs 512 steps of arithmetic", batch=10, trials=9)
    d_all_dog = cp.ones_like(d_mask)
    heavy["portrait condition, all-dog mask"] = measure(
        "heavy_all_dog", "heavy branch: portrait, all-dog mask",
        lambda: heavy_k(grid16, block16, (d_gray, d_all_dog, d_heavy, W_, H_, np.int32(1))),
        note="portrait mode's condition with a mask that is dog everywhere: the same read, no warp disagrees",
        batch=10, trials=9)
    print(f"  diverged warps: {divergence}", flush=True)
    cp.cuda.Device().synchronize()

    return dict(device=dev, kernels=results, checks=checks, divergence=divergence, heavy_branch_us=heavy,
                showcase=showcase, launch_floor_us=round(floor_us, 3), pngs=pngs,
                seconds_on_gpu=round(time.time() - t_start, 1),
                measured_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))


# ===================================================================================================== on your laptop
def fmt_time(us):
    return f"{us:,.1f} µs" if us < 1000 else f"{us / 1000:,.2f} ms"


@app.local_entrypoint()
def main():
    dog = HERE / "dog"
    blog = {name: (dog / "blog" / name).read_bytes() for name in ("gray.png", "checker.png", "portrait.png")}
    r = run.remote((dog / "dog.png").read_bytes(), (dog / "mask.png").read_bytes(), blog)
    out, data = HERE / "out", HERE / "data"
    out.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    for name, b in r.pop("pngs").items():
        (out / f"{name}.png").write_bytes(b)

    d = r["device"]
    kernels = {k["id"]: k for k in r["kernels"]}
    print(f"\n{d['name']} ({d['sms']} SMs), measured {r['measured_at']}")

    print("\nwhat the dog costs, one dog per launch:")
    for key in ("rgb_to_gray", "portrait", "checkerboard"):
        print(f"  {kernels[key]['name']:34s} {fmt_time(kernels[key]['time_us']):>10s}")
    print(f"  {'an empty kernel':34s} {fmt_time(r['launch_floor_us']):>10s}   (no kernel can take less)")
    s = r["showcase"]
    print(f"\npixel ({s['pixel'][0]}, {s['pixel'][1]}): RGB {s['rgb']} -> gray {s['gray']} (the blog: 36)")

    print("\nchecks against the blog's own images:")
    for name, c in r["checks"].items():
        print(f"  {name:18s} {c['identical_pixels']:,} of {c['pixels']:,} pixels identical to {c['against']}"
              + (f" (largest difference: {c['max_difference']})" if c["identical_pixels"] < c["pixels"] else ""))

    print(f"\ndivergence: warps whose 32 threads disagree, counted on the GPU with __ballot_sync, "
          f"of {(W // 16) * (H // 16) * 8:,}:")
    for label, n in r["divergence"].items():
        print(f"  {label:34s} {n:>10,}")
    print("\ndivergence: heavy branches, 512 steps of arithmetic a side:")
    for label, us in r["heavy_branch_us"].items():
        print(f"  {label:34s} {fmt_time(us):>10s}")
    print(f"\nGPU time inside the function: {r['seconds_on_gpu']} s. Images: {out}. Numbers: {data / 'dog_kernels.json'}")

    r["what"] = ("Lecture 3 of the Kernel Engineering Workshop: the kernels of the blog 'The CUDA Programming Model' "
                 "(vizuaraai.github.io/cuda-programming-model), compiled with NVRTC and launched with CuPy on one "
                 "NVIDIA H100 rented on Modal (app l3-dog), on the blog's own 1024 x 768 dog.")
    r["method"] = {"time": "CUDA events around a batch of back-to-back launches queued behind a 1 ms spin kernel; the "
                           "median of 9 to 15 batches after one warm-up launch; microseconds per launch",
                   "launch_floor_us": r["launch_floor_us"]}
    (data / "dog_kernels.json").write_text(json.dumps(r, indent=1))
