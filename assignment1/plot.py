"""Draw the figures for your report from the results a1.py saved. Runs on your laptop; no GPU needed.

    pip install -r requirements.txt
    python plot.py                     # reads results/*.json and writes figures/*.png

It draws only what it finds: run `modal run a1.py::ceilings` first, and the others when you reach them.

Two functions below are yours to complete: arithmetic_intensity(), from your answer to Problem 8(a), and
strided_prediction(), from Problem 9(a)(i). Until you complete them, plot.py skips the roofline figure and draws the
strided figure without your prediction, and says so. You may change anything else here too: the figures are a starting
point for your report, not a required format.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
DATASHEET = {"bf16_tflops": 989.0, "fp32_tflops": 67.0, "hbm_tbps": 3.35}    # H100 SXM, from Lectures 1 and 3


def arithmetic_intensity(m, k, n):
    """Operations per byte of HBM traffic for Y = X @ W, with X of m x k, W of k x n and Y of m x n, all in bf16.

    Problem 8(a): count the operations, count the bytes (each matrix read or written once), and return the ratio.
    """
    return None   # replace this line with your formula


def strided_prediction(stride, stride1_tbps):
    """Useful TB/s that your sector counts predict for a stride, if memory moves bytes as fast as the stride-1 run did.

    Problem 9(a)(i): each thread copies one float (4 bytes read, 4 written, 8 useful bytes). Count the bytes memory
    really moves per float at this stride, from your Problem 4 answers, and scale stride1_tbps by the ratio.
    """
    return None   # replace this line with your formula


def load(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def stamp(ax, facts):
    ax.text(0.99, 0.01, f"{facts['gpu']}, run {facts['run_id']}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="0.5")


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  figures/{name}.png")


def measured_roofs(ceilings):
    """Your GPU's measured ceilings: the cold copy of the largest buffer, and the best bf16 matmul."""
    hbm = max(ceilings["copy"], key=lambda c: c["bytes"])["cold_TBps"]
    bf16 = max(m["TFLOPs"] for m in ceilings["matmul"])
    return hbm, bf16, ceilings["fp32_fma"]["TFLOPs"]


def fig_copy(c):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    mib = [x["bytes"] / 2**20 for x in c["copy"]]
    ax.plot(mib, [x["warm_TBps"] for x in c["copy"]], "o-", label="warm: L2 may still hold the buffers")
    ax.plot(mib, [x["cold_TBps"] for x in c["copy"]], "s-", label="cold: L2 flushed before each copy")
    ax.axhline(DATASHEET["hbm_tbps"], color="0.5", ls="--", lw=1, label="HBM datasheet, 3.35 TB/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("buffer size (MiB); the copy reads one buffer and writes another")
    ax.set_ylabel("bytes read + written per second (TB/s)")
    ax.set_title("Problem 7: copy speed against buffer size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    stamp(ax, c["facts"])
    save(fig, "p7_copy")


def fig_roofline(c, r):
    if arithmetic_intensity(1, 1, 1) is None:
        print("  skipped the roofline figure: complete arithmetic_intensity() in plot.py first (Problem 8a)")
        return
    hbm, bf16, fp32 = measured_roofs(c)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = [10 ** (e / 20) for e in range(-40, 81)]                       # 0.01 to 10,000 FLOP per byte
    for (bw, peak, fp, style, label) in (
            (DATASHEET["hbm_tbps"], DATASHEET["bf16_tflops"], DATASHEET["fp32_tflops"], "--", "datasheet"),
            (hbm, bf16, fp32, "-", "measured")):
        ax.plot(xs, [min(peak, bw * x) for x in xs], style, color="C0", lw=1.2, label=f"bf16 roof, {label}")
        ax.plot(xs, [min(fp, bw * x) for x in xs], style, color="C1", lw=1.0, label=f"fp32 roof, {label}")
    ai = [arithmetic_intensity(row["m"], r["k"], r["n"]) for row in r["rows"]]
    ax.scatter(ai, [row["TFLOPs"] for row in r["rows"]], s=14, color="C3", zorder=3, label="your matmuls")
    for row, x in zip(r["rows"], ai):
        if row["m"] in (1, 8, 64, 256, 1024, 8192):
            ax.annotate(f"M={row['m']}", (x, row["TFLOPs"]), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 1e4)
    ax.set_ylim(0.01, 2000)
    ax.set_xlabel("arithmetic intensity (FLOP per byte of HBM traffic)")
    ax.set_ylabel("TFLOP/s")
    ax.set_title(f"Problem 8: Y = X @ W, W of {r['k']} x {r['n']}, on the roofline")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper left")
    stamp(ax, r["facts"])
    save(fig, "p8_roofline")


def fig_fine(r):
    rows = [row for row in r["rows"] if row["m"] in r["fine"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot([row["m"] for row in rows], [row["us"] for row in rows], "o-")
    ax.set_xlabel("M (tokens in the batch)")
    ax.set_ylabel("time (us)")
    ax.set_title("Problem 8(c): a fine sweep of batch sizes")
    ax.grid(alpha=0.3)
    stamp(ax, r["facts"])
    save(fig, "p8_fine")


def fig_strided(m):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    s = [row["stride"] for row in m["strided"]]
    ax.plot(s, [row["useful_TBps"] for row in m["strided"]], "o-", label="measured")
    one = [row["useful_TBps"] for row in m["strided"] if row["stride"] == 1][0]
    if strided_prediction(1, one) is None:
        print("  drew the strided figure without your prediction: complete strided_prediction() (Problem 9a)")
    else:
        ax.plot(s, [strided_prediction(x, one) for x in s], "s--", label="your sector model")
    ax.set_xscale("log", base=2)
    ax.set_xticks(s, [str(x) for x in s])
    ax.set_xlabel("stride (floats between the reads of neighbouring threads)")
    ax.set_ylabel("useful TB/s (8 bytes a float: read + write)")
    ax.set_title("Problem 9(a): strided reads")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    stamp(ax, m["facts"])
    save(fig, "p9_strided")


def fig_gray(m):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, block in enumerate(sorted({row["block"] for row in m["gray"]})):
        rows = sorted((row for row in m["gray"] if row["block"] == block), key=lambda x: x["bytes_per_thread"])
        ax.plot([row["bytes_per_thread"] + 0.4 * (i - 1.5) for row in rows], [row["TBps"] for row in rows], "o",
                label=f"{block} threads a block")
    ax.axhline(DATASHEET["hbm_tbps"], color="0.5", ls="--", lw=1, label="HBM datasheet, 3.35 TB/s")
    ax.set_xscale("log", base=2)
    ticks = sorted({row["bytes_per_thread"] for row in m["gray"]})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("bytes each thread loads at once (4 x VEC x ILP)")
    ax.set_ylabel("TB/s (4 bytes in + 1 byte out a pixel)")
    ax.set_title("Problem 9(b): grayscale of 64 dogs, bytes in flight")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    stamp(ax, m["facts"])
    save(fig, "p9_gray")


def fig_rope(rp):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    s = [row["S"] for row in rp["gpu"]]
    for key, label in (("copy_us", "copy of the same bytes"), ("eager_us", "eager PyTorch"),
                       ("compiled_us", "torch.compile"), ("cuda_us", f"CUDA kernel, {rp['cuda_pairs']} pairs a thread")):
        ax.plot(s, [row[key] for row in rp["gpu"]], "o-", label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("S (tokens)")
    ax.set_ylabel("GPU time per call (us), L2 flushed")
    ax.set_title("Problem 10: RoPE on q and k of Qwen3-8B")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    stamp(ax, rp["facts"])
    save(fig, "p10_rope")


def main():
    c, r, m, rp = load("ceilings"), load("roofline"), load("memory"), load("rope")
    if not any((c, r, m, rp)):
        print("no results yet: run `modal run a1.py::ceilings` first")
        return
    print("writing:")
    if c:
        fig_copy(c)
    if r:
        fig_fine(r)
        if c:
            fig_roofline(c, r)
        else:
            print("  skipped the roofline figure: it needs results/ceilings.json too")
    if m:
        fig_strided(m)
        fig_gray(m)
    if rp:
        fig_rope(rp)


if __name__ == "__main__":
    main()
