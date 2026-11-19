"""Assignment 2 practical: the matmul ladder and autotuning, measured on one H100.

    pip install -r requirements.txt
    modal setup                          # once, if you have not set up Modal for Assignment 1
    python my_element.py you@email.com   # your own element of C, for Problems 1 and 7 (no GPU)
    modal run a2.py::check               # are your kernels right? small and real shapes, no timing
    modal run a2.py::ladder              # Problem 6   every rung at 128, 1,024 and 8,192 tokens, and cuBLAS
    modal run a2.py::sweep --knob tm     # Problems 8 to 11, one knob at a time: tm, patch, tile, bk, warp (or all)
    modal run a2.py::tune                # Problem 12  every configuration your tune.py allows, at all three shapes
    modal run a2.py::final               # Problem 13  your best.json, timed carefully against cuBLAS
    python plot.py                       # draws your figures from results/*.json, on your laptop

The product is the one Problem 8 of Assignment 1 timed with cuBLAS: Qwen3-8B's MLP up-projection Y = X W, with X of
M x 4,096 (M tokens) and W of 4,096 x 12,288. Here it runs in 32-bit decimals (FP32) on the CUDA cores, the way
Lectures 7 and 9 ran it, for M = 128, 1,024 and 8,192. In the code, A is X (M x K), B is W (K x N) and C is Y (M x N),
all stored row after row, with K = 4,096 and N = 12,288.

Your kernels live in kernels/, one file for each rung, each a CUDA C kernel named sgemm. The sizes a kernel takes (BM,
BN, BK, TM, ...) arrive as -D flags when it is compiled, so one source compiles to every configuration; launch shapes
are fixed by this file (FAMILIES below), so your kernels must follow them. Do not change how this file times or
checks: we rerun your best.json with our own copy.

Every time comes from CUDA events around a batch of back-to-back launches queued behind a 1 ms spin, the median of
several batches (the method of Lecture 3's dog.py and Lecture 7's sgemm.py). The L2 is not flushed: W alone is 201 MB,
four times the L2. Every result is checked against the same product computed in 64-bit decimals.
"""
import json
import os
import sys
import time
from pathlib import Path

import modal

os.environ.setdefault("CUPY_TF32", "0")     # cuBLAS must do true FP32 arithmetic too, not TF32 on the tensor cores

HERE = Path(__file__).resolve().parent
app = modal.App("a2-matmul-ladder")
# The same image as Assignment 1, pinned by digest and every package by version and hash, so if you ran Assignment 1
# Modal reuses it and nothing is built.
image = (modal.Image.from_registry(
             "nvidia/cuda:13.0.0-devel-ubuntu24.04"
             "@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d", add_python="3.12")
         .entrypoint([])
         .pip_install_from_requirements(str(HERE / "image-requirements.txt")))

with image.imports():
    import cupy as cp
    import numpy as np

K_DIM, N_DIM = 4096, 12288
SHAPES = [128, 1024, 8192]                  # M, the number of tokens
SMEM_LIMIT = 48 * 1024                      # static shared memory a block may declare
REG_LIMIT = 255

# ===================================================================================================== the rungs
# name -> file, the sizes it takes, its default sizes (the lectures' ladder), threads a block, and the grid.
FAMILIES = {
    "k1": dict(file="k1_naive.cu", params=[], default={}),
    "k2": dict(file="k2_coalesced.cu", params=[], default={}),
    "k3": dict(file="k3_smem.cu", params=[], default={}),
    "k4": dict(file="k4_1d.cu", params=["BM", "BN", "BK", "TM"], default=dict(BM=64, BN=64, BK=8, TM=8)),
    "k5": dict(file="k5_2d.cu", params=["BM", "BN", "BK", "TM", "TN"], default=dict(BM=128, BN=128, BK=8, TM=8, TN=8)),
    "k6": dict(file="k6_vec.cu", params=["BM", "BN", "BK", "TM", "TN"], default=dict(BM=128, BN=128, BK=8, TM=8, TN=8)),
    "k7": dict(file="k7_warptile.cu", params=["BM", "BN", "BK", "WM", "WN", "WNITER", "TM", "TN"],
               default=dict(BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4)),
}


def threads(fam, p):
    if fam in ("k1", "k2", "k3"):
        return 1024
    if fam == "k4":
        return p["BM"] * p["BN"] // p["TM"]
    if fam in ("k5", "k6"):
        return (p["BM"] // p["TM"]) * (p["BN"] // p["TN"])
    return 32 * (p["BM"] // p["WM"]) * (p["BN"] // p["WN"])


def geometry(fam, p, M):
    """(grid, block) for one launch with M tokens."""
    if fam == "k1":
        return (M // 32, N_DIM // 32, 1), (32, 32, 1)
    if fam in ("k2", "k3"):
        return (N_DIM // 32, M // 32, 1), (32, 32, 1)
    return (N_DIM // p["BN"], M // p["BM"], 1), (threads(fam, p), 1, 1)


def broken_rule(fam, p, M=None):
    """The first hard rule these sizes break, or None. These are the rules without which a kernel cannot compile, cannot
    launch, or reads outside its matrices. (Too many registers is not one of them: the compiler spills, and the kernel
    is slow but right. Your own check() in tune.py should catch that too.)"""
    if fam in ("k1", "k2", "k3"):
        return None if M is None or M % 32 == 0 else "divides"
    for k in FAMILIES[fam]["params"]:
        if not isinstance(p.get(k), int) or p[k] < 1:
            return "missing"
    nt = threads(fam, p) if all(p[k] for k in FAMILIES[fam]["params"]) else 0
    if fam == "k4":
        if p["BM"] % p["TM"]:
            return "divides"
    elif fam in ("k5", "k6"):
        if p["BM"] % p["TM"] or p["BN"] % p["TN"]:
            return "divides"
    else:
        if p["BM"] % p["WM"] or p["BN"] % p["WN"] or p["WN"] % p["WNITER"]:
            return "divides"
        per = 32 * p["TM"] * p["TN"] * p["WNITER"]
        if (p["WM"] * p["WN"]) % per or p["WM"] * p["WN"] < per:
            return "warp"
        wmiter = p["WM"] * p["WN"] // per
        if p["WM"] % wmiter:
            return "warp"
        wsubm, wsubn = p["WM"] // wmiter, p["WN"] // p["WNITER"]
        if wsubm % p["TM"] or wsubn % p["TN"] or (wsubm // p["TM"]) * (wsubn // p["TN"]) != 32:
            return "warp"
    if not (32 <= nt <= 1024 and nt % 32 == 0):
        return "threads"
    if (p["BM"] * p["BK"] + p["BK"] * p["BN"]) * 4 > SMEM_LIMIT:
        return "smem"
    if fam in ("k6", "k7"):
        if p["BK"] % 4 or p["BN"] % 4 or p["BM"] % 4 or p["TN"] % 4 or (fam == "k6" and p["TM"] % 4):
            return "vector"
    if M is not None and (M % p["BM"] or N_DIM % p["BN"] or K_DIM % p["BK"]):
        return "divides"
    return None


def cid(fam, p):
    """A short name for one configuration, e.g. k6_128x64_bk16_8x4 or k7_128x128_bk16_w64x64_it4_8x4."""
    if fam in ("k1", "k2", "k3"):
        return fam
    if fam == "k4":
        return f"k4_{p['BM']}x{p['BN']}_bk{p['BK']}_tm{p['TM']}"
    if fam in ("k5", "k6"):
        return f"{fam}_{p['BM']}x{p['BN']}_bk{p['BK']}_{p['TM']}x{p['TN']}"
    return f"k7_{p['BM']}x{p['BN']}_bk{p['BK']}_w{p['WM']}x{p['WN']}_it{p['WNITER']}_{p['TM']}x{p['TN']}"


# ===================================================================================================== the sweeps
def sweep_configs(knob):
    """The one-knob sweeps of Problems 8 to 11: everything else held still. Every student runs the same ones."""
    out = []
    if knob == "tm":            # Problem 8: elements of a column a thread owns, in the 1D kernel
        for bm, bk in ((32, 32), (64, 8)):
            for tm in (1, 2, 4, 8, 16, 32, 64):
                out.append(("k4", dict(BM=bm, BN=bm, BK=bk, TM=tm)))
    elif knob == "patch":       # Problem 9: the thread's patch, in a 128 x 128 tile
        for fam in ("k5", "k6"):
            for tm, tn in ((4, 4), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16)):
                out.append((fam, dict(BM=128, BN=128, BK=8, TM=tm, TN=tn)))
    elif knob == "tile":        # Problem 10: the block's tile, with two patch sizes
        for tm in (4, 8):
            for bm in (32, 64, 128, 256):
                for bn in (32, 64, 128, 256):
                    out.append(("k6", dict(BM=bm, BN=bn, BK=16, TM=tm, TN=tm)))
    elif knob == "bk":          # Problem 11(a): the chunk
        for bm, t in ((64, 4), (128, 8)):
            for bk in (4, 8, 16, 32, 64):
                out.append(("k6", dict(BM=bm, BN=bm, BK=bk, TM=t, TN=t)))
    elif knob == "warp":        # Problem 11(b): the warp's tile, in a 128 x 128 tile with 8 x 4 patches
        for wm, wn in ((16, 128), (32, 64), (64, 32), (128, 16), (32, 128), (64, 64), (128, 32),
                       (16, 64), (32, 32), (64, 16)):
            for it in (1, 2, 4):
                out.append(("k7", dict(BM=128, BN=128, BK=16, WM=wm, WN=wn, WNITER=it, TM=8, TN=4)))
    else:
        raise ValueError(f"no knob called {knob!r}; choose tm, patch, tile, bk, warp or all")
    return out


KNOBS = ["tm", "patch", "tile", "bk", "warp"]


# ===================================================================================================== on the GPU
@app.function(image=image, gpu="H100!", max_containers=1, timeout=3000, cpu=16.0, memory=32768)
def measure_many(sources: dict, configs: list, shapes: list, mode: str = "sweep", seed: int = 5) -> dict:
    """Compile every configuration, check it against FP64 at each shape, and time it. configs: [[family, params]]."""
    import subprocess
    import threading

    def watchdog(seconds=2940):                 # whatever happens, the container stops before Modal's timeout
        time.sleep(seconds)
        print(f"watchdog: {seconds} s passed, stopping", flush=True)
        os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()

    t_start = time.time()
    props = cp.cuda.runtime.getDeviceProperties(0)
    q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    driver, sm_mhz, watts = [x.strip() for x in (q.split(",") + ["", "", ""])[:3]]
    facts = dict(gpu=props["name"].decode(), sms=props["multiProcessorCount"], cc=f'{props["major"]}.{props["minor"]}',
                 driver=driver, max_sm_clock_mhz=sm_mhz, power_limit_w=watts, cupy=cp.__version__,
                 cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cupy_tf32=os.environ.get("CUPY_TF32"),
                 run_id=modal.current_function_call_id(), utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    print(f"{facts['gpu']}: {facts['sms']} SMs, driver {driver}, run {facts['run_id']}", flush=True)

    spin = cp.RawKernel(r'''
extern "C" __global__ void spin(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}''', "spin")

    big = mode == "final"

    def gpu_us(launch):
        """Median microseconds per launch: CUDA events around a batch of launches queued behind a 1 ms spin."""
        target_ms, trials = (40.0, 15) if big else (10.0, 5)
        launch()
        cp.cuda.Device().synchronize()
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        launch()
        e.record()
        e.synchronize()
        one = cp.cuda.get_elapsed_time(s, e)
        batch = int(max(1, min(200, target_ms / max(one, 1e-3))))
        if one > 200:
            trials = 3
        out = []
        for _ in range(trials):
            spin((1,), (1,), (np.uint64(1_000_000),))
            s.record()
            for _ in range(batch):
                launch()
            e.record()
            e.synchronize()
            out.append(cp.cuda.get_elapsed_time(s, e) * 1e3 / batch)
        return float(np.median(out)), float(np.min(out)), float(np.max(out))

    def blocks_per_sm(fn, nt):
        try:
            return int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(fn.ptr, nt, 0))
        except AttributeError:
            import ctypes
            lib, nb = ctypes.CDLL("libcuda.so.1"), ctypes.c_int()
            lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(nb), ctypes.c_void_p(fn.ptr),
                                                            ctypes.c_int(nt), ctypes.c_size_t(0))
            return nb.value

    # inputs and their FP64 products
    rng = np.random.default_rng(seed)
    B = cp.asarray(rng.uniform(-1, 1, (K_DIM, N_DIM)).astype(np.float32))
    data = {}
    for M in shapes:
        A = cp.asarray(rng.uniform(-1, 1, (M, K_DIM)).astype(np.float32))
        ref = cp.matmul(A.astype(cp.float64), B.astype(cp.float64))
        data[M] = dict(A=A, C=cp.empty((M, N_DIM), cp.float32), ref=ref, refmax=float(cp.abs(ref).max()))

    def rel_err(M, C):
        return float(cp.abs(C.astype(cp.float64) - data[M]["ref"]).max()) / data[M]["refmax"]

    cublas = {}
    for M in shapes:
        d = data[M]
        cp.matmul(d["A"], B, out=d["C"])
        err = rel_err(M, d["C"])
        med, lo, hi = gpu_us(lambda: cp.matmul(d["A"], B, out=d["C"]))
        cublas[str(M)] = dict(time_us=round(med, 2), time_us_min=round(lo, 2), time_us_max=round(hi, 2),
                              tflops=round(2 * M * N_DIM * K_DIM / med / 1e6, 3), err_rel=err)
        print(f"cuBLAS FP32, M={M:5d}: {med:10.1f} us {cublas[str(M)]['tflops']:6.2f} TFLOP/s  rel err {err:.1e}",
              flush=True)

    # Compile every configuration first, many at once: nvcc runs in parallel on the machine's CPU cores while the GPU
    # waits, which is several times faster than compiling them one by one.
    from concurrent.futures import ThreadPoolExecutor
    arch = f"sm_{props['major']}{props['minor']}"
    build = "/tmp/a2build"
    os.makedirs(build, exist_ok=True)
    for fam, src in sources.items():
        with open(f"{build}/{fam}.cu", "w") as f:
            f.write(src)

    def nvcc(job):
        i, fam, p = job
        if broken_rule(fam, p):
            return i, None, None
        out = f"{build}/c{i}.cubin"
        cmd = ["nvcc", "-cubin", f"-arch={arch}", "-O3", "-std=c++17",
               *[f"-D{k}={p[k]}" for k in FAMILIES[fam]["params"]], "-o", out, f"{build}/{fam}.cu"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return i, None, "compile: nvcc took more than 10 minutes"
        return i, (out if r.returncode == 0 else None), (None if r.returncode == 0 else
                                                          "compile: " + (r.stderr or r.stdout)[-1500:])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 4)) as pool:
        built = {i: (out, err) for i, out, err in pool.map(nvcc, [(i, f, p) for i, (f, p) in enumerate(configs)])}
    print(f"compiled {len(configs)} configurations in {time.time() - t0:.0f} s", flush=True)

    budget = 2700 if mode in ("tune", "sweep") else 2800   # seconds; past this, return what is measured so far
    records, dead = [], None
    for i, (fam, p) in enumerate(configs):
        rec = dict(id=cid(fam, p), family=fam, params=p, results={})
        records.append(rec)
        if dead:
            rec["error"] = "not run: the GPU context was lost earlier (see the record marked crashed)"
            continue
        rule = broken_rule(fam, p)
        if rule:
            rec["rejected"] = rule
            continue
        if time.time() - t_start > budget:
            rec["error"] = f"not run: out of time ({budget} s); run fewer configurations at once"
            continue
        cubin, err = built[i]
        if not cubin:
            rec["error"] = err
            print(f"[{i + 1}/{len(configs)}] {rec['id']}: does not compile", flush=True)
            continue
        try:
            k = cp.RawModule(path=cubin).get_function("sgemm")
            nt = threads(fam, p)
            rec.update(threads=nt, regs=int(k.num_regs), spill_bytes=int(k.local_size_bytes),
                       smem_bytes=int(k.shared_size_bytes), blocks_per_sm=blocks_per_sm(k.kernel, nt))
        except Exception as ex:
            rec["error"] = f"load: {type(ex).__name__}: {ex}"[:1500]
            continue
        line = []
        for M in shapes:
            if broken_rule(fam, p, M):
                rec["results"][str(M)] = dict(skipped="the tile does not divide this shape")
                continue
            grid, block = geometry(fam, p, M)
            d = data[M]
            launch = lambda: k(grid, block, (d["A"], B, d["C"], np.int32(M), np.int32(N_DIM), np.int32(K_DIM)))
            try:
                d["C"].fill(float("nan"))
                launch()
                cp.cuda.Device().synchronize()
                err = rel_err(M, d["C"])
                ok = bool(err < 1e-4)
                r = dict(blocks=grid[0] * grid[1], err_rel=err, correct=ok)
                if ok and mode != "check":
                    med, lo, hi = gpu_us(launch)
                    r.update(time_us=round(med, 2), time_us_min=round(lo, 2), time_us_max=round(hi, 2),
                             tflops=round(2 * M * N_DIM * K_DIM / med / 1e6, 3),
                             pct_cublas=round(100 * cublas[str(M)]["time_us"] / med, 1))
                    line.append(f"M={M}: {med:9.1f} us {r['pct_cublas']:5.1f}%")
                elif ok:
                    line.append(f"M={M}: right")
                else:
                    line.append(f"M={M}: WRONG (rel err {err:.1e})")
                rec["results"][str(M)] = r
            except Exception as ex:
                rec["crashed"] = f"M={M}: {type(ex).__name__}: {ex}"[:600]
                dead = rec["id"]
                print(f"[{i + 1}/{len(configs)}] {rec['id']}: CRASHED at M={M}: {ex}; stopping here", flush=True)
                break
        if not dead:
            print(f"[{i + 1}/{len(configs)}] {rec['id']:40s} regs {rec['regs']:3d} spill {rec['spill_bytes']:4d} "
                  f"blk/SM {rec['blocks_per_sm']:2d} | " + " | ".join(line), flush=True)
    facts["seconds_on_gpu"] = round(time.time() - t_start, 1)
    return dict(facts=facts, shapes=shapes, K=K_DIM, N=N_DIM, mode=mode, seed=seed, cublas=cublas, records=records,
                crashed=dead)


# ===================================================================================================== on your laptop
def read_sources(kernels: str) -> dict:
    folder = Path(kernels) if kernels else HERE / "kernels"
    if not folder.is_absolute():
        folder = HERE / folder
    return {fam: (folder / f["file"]).read_text() for fam, f in FAMILIES.items() if (folder / f["file"]).exists()}


def save(name, result, extra=None):
    result["facts"]["app_id"] = app.app_id          # the ap-... ID that modal.com/apps shows for this run
    if extra:
        result.update(extra)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{name}.json"
    if path.exists():                               # never overwrite an earlier run: keep it as name_runN.json
        i = 2
        while (out / f"{name}_run{i}.json").exists():
            i += 1
        path.rename(out / f"{name}_run{i}.json")
    path.write_text(json.dumps(result, indent=1))
    f = result["facts"]
    print(f"\n{f['gpu']} ({f['sms']} SMs), app {f['app_id']}, run {f['run_id']}, {f['utc']}, "
          f"{f['seconds_on_gpu']} s on the GPU\nsaved {path}")


def table(result, top=None, sort_by=None):
    shapes = result["shapes"]
    recs = [r for r in result["records"] if r.get("results")]
    if sort_by:
        recs = sorted(recs, key=lambda r: r["results"].get(sort_by, {}).get("time_us", 1e18))
    if top:
        recs = recs[:top]
    print(f"\n{'configuration':42s} {'regs':>4s} {'spill':>5s} {'b/SM':>4s}" +
          "".join(f"{'M=' + str(M):>20s}" for M in shapes))
    for r in recs:
        cells = []
        for M in shapes:
            x = r["results"].get(str(M), {})
            if "time_us" in x:
                cells.append(f"{x['time_us']:10.1f} us {x['pct_cublas']:4.0f}%")
            elif x.get("correct") is False:
                cells.append(f"{'WRONG':>18s}")
            else:
                cells.append(f"{'-':>18s}")
        print(f"{r['id']:42s} {r.get('regs', 0):4d} {r.get('spill_bytes', 0):5d} {r.get('blocks_per_sm', 0):4d}" +
              "".join(f"{c:>20s}" for c in cells))
    print(f"{'cuBLAS (FP32)':42s} {'':15s}" + "".join(
        f"{result['cublas'][str(M)]['time_us']:10.1f} us {'100%':>5s}" for M in shapes))
    other = [r for r in result["records"] if not r.get("results")]
    for r in other:
        why = r.get("rejected") and f"breaks the rule '{r['rejected']}'" or r.get("error", "")[:200]
        print(f"  not timed: {r['id']}: {why}")
    if result.get("crashed"):
        print(f"\nThe GPU context was lost at {result['crashed']} (usually a read or write outside the matrices); "
              f"every configuration after it was not run. Fix that kernel and run again.")


@app.local_entrypoint()
def check(kernels: str = ""):
    """Run every kernel in kernels/ once with its default sizes, at every shape, and say whether it is right."""
    src = read_sources(kernels)
    configs = [[fam, FAMILIES[fam]["default"]] for fam in FAMILIES if fam in src]
    r = measure_many.remote(src, configs, SHAPES, mode="check")
    for rec in r["records"]:
        res = rec.get("results", {})
        verdict = ", ".join(f"M={M}: {'right' if v.get('correct') else 'WRONG'}" for M, v in res.items()) or \
            rec.get("error", rec.get("rejected", ""))[:300]
        print(f"{rec['id']:44s} {verdict}")
    missing = [fam for fam in FAMILIES if fam not in src]
    if missing:
        print("no file yet for: " + ", ".join(FAMILIES[f]["file"] for f in missing))
    save("check", r)


@app.local_entrypoint()
def ladder(kernels: str = ""):
    """Problem 6: every rung with the lectures' sizes, at 128, 1,024 and 8,192 tokens, against cuBLAS."""
    src = read_sources(kernels)
    configs = [[fam, FAMILIES[fam]["default"]] for fam in FAMILIES if fam in src]
    r = measure_many.remote(src, configs, SHAPES, mode="ladder")
    table(r)
    save("ladder", r)


@app.local_entrypoint()
def sweep(knob: str = "all", kernels: str = ""):
    """Problems 8 to 11: one knob at a time. --knob tm | patch | tile | bk | warp | all"""
    src = read_sources(kernels)
    knobs = KNOBS if knob == "all" else [knob]
    configs = [[fam, p] for kn in knobs for fam, p in sweep_configs(kn) if fam in src]
    r = measure_many.remote(src, configs, SHAPES, mode="sweep")
    table(r)
    save(f"sweep_{knob}", r, dict(knobs=knobs))


@app.local_entrypoint()
def tune(kernels: str = "", tuner: str = "tune.py", limit: int = 600):
    """Problem 12: every configuration tune.py's space() yields and its check() allows, at all three shapes."""
    import importlib.util
    from collections import Counter

    spec = importlib.util.spec_from_file_location("student_tune", HERE / tuner if not Path(tuner).is_absolute()
                                                  else tuner)
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    src = read_sources(kernels)
    space = [(fam, dict(p)) for fam, p in tm.space()]
    pruned, keep, seen = Counter(), [], set()
    for fam, p in space:
        if (fam, cid(fam, p)) in seen:
            continue
        seen.add((fam, cid(fam, p)))
        why = tm.check(fam, p)
        if why:
            pruned[why] += 1
        else:
            keep.append([fam, p])
    print(f"space(): {len(space)} configurations ({len(seen)} different); your check() removed "
          f"{sum(pruned.values())}: " + ", ".join(f"{k} {v}" for k, v in pruned.most_common()))
    slipped = [(fam, p, broken_rule(fam, p)) for fam, p in keep if broken_rule(fam, p)]
    if slipped:
        print(f"{len(slipped)} that your check() kept break a hard rule and will not be timed, for example "
              f"{cid(slipped[0][0], slipped[0][1])} ({slipped[0][2]})")
    if len(keep) > limit:
        print(f"{len(keep)} configurations is more than --limit {limit}; tighten check() or raise the limit "
              f"(every 100 cost very roughly 15 cents)")
        return
    print(f"timing {len(keep)} configurations at M = {SHAPES}")
    r = measure_many.remote(src, keep, SHAPES, mode="tune")
    for M in SHAPES:
        print(f"\nfastest ten at M = {M}:")
        table(dict(r, records=[x for x in r["records"] if x.get("results")]), top=10, sort_by=str(M))
    save("tune", r, dict(space_size=len(space), pruned=dict(pruned), tuner=Path(tuner).name))


@app.local_entrypoint()
def final(kernels: str = "", best: str = "best.json"):
    """Problem 13: the configuration best.json names for each shape, on new random inputs, timed with more launches."""
    choice = json.loads((HERE / best).read_text())
    src = read_sources(kernels)
    configs = []
    for M in SHAPES:
        c = choice[str(M)]
        configs.append([c["family"], c["params"]])
    r = measure_many.remote(src, configs, SHAPES, mode="final", seed=int(time.time()) % 100000)
    print(f"\n{'M':>6s}  {'your configuration':44s} {'time':>12s} {'TFLOP/s':>8s} {'cuBLAS':>10s} {'share':>6s}")
    for M, rec in zip(SHAPES, r["records"]):
        x = rec.get("results", {}).get(str(M), {})
        cb = r["cublas"][str(M)]
        if "time_us" in x:
            print(f"{M:6d}  {rec['id']:44s} {x['time_us']:9.1f} us {x['tflops']:8.2f} {cb['time_us']:7.1f} us "
                  f"{x['pct_cublas']:5.1f}%")
        else:
            print(f"{M:6d}  {rec['id']:44s} not timed: {x or rec.get('error') or rec.get('rejected')}")
    save("final", r, dict(best=choice))
