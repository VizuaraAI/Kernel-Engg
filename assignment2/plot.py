"""Draws the figures for your report from results/*.json, on your laptop, and prints the tables Problems 8 to 12 ask
for (the regret matrix among them).

    python plot.py                  # every figure it has the results for, into figures/
    python plot.py --results DIR    # read the JSON files from another folder

Each figure says which run (app ID) it comes from. Nothing here is graded; the explanations are.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SHAPES = [128, 1024, 8192]
COLORS = {128: "#B0710F", 1024: "#177245", 8192: "#3B5BA5"}


def load(folder, name):
    p = folder / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def sweeps(folder):
    """Every one-knob record from sweep_*.json (newest file of each name), by configuration id."""
    recs, apps = {}, set()
    for p in sorted(folder.glob("sweep_*.json")):
        if "_run" in p.stem:
            continue
        r = json.loads(p.read_text())
        apps.add(r["facts"]["app_id"])
        for x in r["records"]:
            recs[x["id"]] = dict(x, cublas=r["cublas"])
    return recs, ", ".join(sorted(apps))


def t(rec, M):
    return rec.get("results", {}).get(str(M), {}).get("time_us")


def stamp(fig, app):
    fig.text(0.99, 0.005, f"run {app}", ha="right", va="bottom", fontsize=7, color="#615C50")


def ladder(folder, out):
    r = load(folder, "ladder")
    if not r:
        return
    recs = [x for x in r["records"] if x.get("results")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.26
    for i, M in enumerate(SHAPES):
        ys = [x["results"].get(str(M), {}).get("pct_cublas", 0) for x in recs]
        ax.bar([j + (i - 1) * w for j in range(len(recs))], ys, w, color=COLORS[M], label=f"M = {M:,} tokens")
    ax.set_xticks(range(len(recs)), [x["family"] for x in recs])
    ax.set_ylabel("speed, % of cuBLAS FP32")
    ax.set_title("The ladder with the lectures' sizes")
    ax.axhline(100, color="#615C50", lw=0.8, ls=":")
    ax.legend(frameon=False)
    stamp(fig, r["facts"]["app_id"])
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=160)
    print(f"\nLadder (% of cuBLAS), run {r['facts']['app_id']}")
    for x in recs:
        print(f"  {x['id']:40s}" + "".join(f"{x['results'].get(str(M), {}).get('pct_cublas', 0):8.1f}" for M in SHAPES))


def line_knob(recs, app, out, name, series, xlabel, xvals, title):
    """series: list of (label, [config ids in x order]). Plots time / fastest time at each shape."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, M in zip(axes, SHAPES):
        best = min((t(recs[i], M) for _, ids in series for i in ids if i in recs and t(recs[i], M)), default=None)
        for (label, ids), ls in zip(series, ("-", "--", ":")):
            xs = [x for x, i in zip(xvals, ids) if i in recs and t(recs[i], M)]
            ys = [t(recs[i], M) / best for i in ids if i in recs and t(recs[i], M)]
            ax.plot(xs, ys, ls, marker="o", color=COLORS[M], label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xvals, [str(v) for v in xvals])
        ax.set_title(f"M = {M:,}")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("time ÷ fastest in this figure")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    stamp(fig, app)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=160)


def knob_table(recs, ids, title):
    print(f"\n{title}")
    print(f"  {'configuration':40s} {'threads':>7s} {'regs':>4s} {'spill':>5s} {'b/SM':>4s}" +
          "".join(f"{'M=' + str(M):>11s}" for M in SHAPES))
    for i in ids:
        x = recs.get(i)
        if not x:
            continue
        if not x.get("results"):
            print(f"  {i:40s} not timed: {x.get('rejected') or x.get('error', '')[:60]}")
            continue
        print(f"  {i:40s} {x.get('threads', 0):7d} {x.get('regs', 0):4d} {x.get('spill_bytes', 0):5d} "
              f"{x.get('blocks_per_sm', 0):4d}" + "".join(
                  f"{t(x, M):9.0f}us" if t(x, M) else f"{'-':>11s}" for M in SHAPES))


def all_sweeps(folder, out):
    recs, app = sweeps(folder)
    if not recs:
        return
    tms = [1, 2, 4, 8, 16, 32, 64]
    s1 = [f"k4_32x32_bk32_tm{v}" for v in tms]
    s2 = [f"k4_64x64_bk8_tm{v}" for v in tms]
    if any(i in recs for i in s1 + s2):
        line_knob(recs, app, out, "sweep_tm", [("32 x 32 tile, chunks of 32", s1), ("64 x 64 tile, chunks of 8", s2)],
                  "TM: elements a thread owns", tms, "Problem 8: the 1D kernel, one knob")
        knob_table(recs, s1 + s2, "Problem 8: TM in the 1D kernel")
    patches = [(4, 4), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16)]
    pids = {fam: [f"{fam}_128x128_bk8_{a}x{b}" for a, b in patches] for fam in ("k5", "k6")}
    if any(i in recs for v in pids.values() for i in v):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, M in zip(axes, SHAPES):
            for k, (fam, ids) in enumerate(pids.items()):
                ys = [recs[i]["results"][str(M)]["pct_cublas"] if i in recs and t(recs[i], M) else 0 for i in ids]
                ax.bar([j + (k - 0.5) * 0.4 for j in range(len(ids))], ys, 0.4, label=f"kernel {fam[1]}",
                       color=COLORS[M], alpha=1.0 if fam == "k6" else 0.45)
            ax.set_xticks(range(len(patches)), [f"{a}x{b}" for a, b in patches], fontsize=8)
            ax.set_title(f"M = {M:,}")
            ax.set_xlabel("patch TM x TN")
        axes[0].set_ylabel("% of cuBLAS")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Problem 9: the thread's patch, in a 128 x 128 tile")
        stamp(fig, app)
        fig.tight_layout()
        fig.savefig(out / "sweep_patch.png", dpi=160)
        knob_table(recs, pids["k5"] + pids["k6"], "Problem 9: the patch")
    sizes = [32, 64, 128, 256]
    for p in (4, 8):
        ids = [f"k6_{bm}x{bn}_bk16_{p}x{p}" for bm in sizes for bn in sizes]
        if not any(i in recs for i in ids):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, M in zip(axes, SHAPES):
            grid = [[(recs.get(f"k6_{bm}x{bn}_bk16_{p}x{p}", {}).get("results", {}).get(str(M), {})
                      .get("pct_cublas", math.nan)) for bn in sizes] for bm in sizes]
            ax.imshow(grid, cmap="Greens", vmin=0, vmax=80)
            for a, row in enumerate(grid):
                for b, v in enumerate(row):
                    ax.text(b, a, "-" if v != v else f"{v:.0f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(4), sizes)
            ax.set_yticks(range(4), sizes)
            ax.set_xlabel("BN")
            ax.set_ylabel("BM")
            ax.set_title(f"M = {M:,}: % of cuBLAS")
        fig.suptitle(f"Problem 10: the block's tile, {p} x {p} patches, chunks of 16")
        stamp(fig, app)
        fig.tight_layout()
        fig.savefig(out / f"sweep_tile_{p}x{p}.png", dpi=160)
        knob_table(recs, ids, f"Problem 10: the block's tile, {p} x {p} patches")
    bks = [4, 8, 16, 32, 64]
    b1 = [f"k6_64x64_bk{v}_4x4" for v in bks]
    b2 = [f"k6_128x128_bk{v}_8x8" for v in bks]
    if any(i in recs for i in b1 + b2):
        line_knob(recs, app, out, "sweep_bk", [("64 x 64 tile, 4 x 4 patches", b1), ("128 x 128 tile, 8 x 8", b2)],
                  "BK: steps of k in a chunk", bks, "Problem 11(a): the chunk")
        knob_table(recs, b1 + b2, "Problem 11(a): the chunk")
    warps = [(16, 128), (32, 64), (64, 32), (128, 16), (32, 128), (64, 64), (128, 32), (16, 64), (32, 32), (64, 16)]
    wids = [f"k7_128x128_bk16_w{a}x{b}_it{i}_8x4" for a, b in warps for i in (1, 2, 4)]
    if any(i in recs for i in wids):
        knob_table(recs, wids, "Problem 11(b): the warp's tile (128 x 128 block tile, 8 x 4 patches, chunks of 16)")


def tune(folder, out):
    r = load(folder, "tune")
    if not r:
        return
    recs = [x for x in r["records"] if x.get("results")]
    best = {}
    for M in SHAPES:
        ok = [x for x in recs if t(x, M)]
        best[M] = min(ok, key=lambda x: t(x, M))
    print(f"\nTune, run {r['facts']['app_id']}: {len(r['records'])} configurations timed or tried")
    print("Regret matrix: the winner at each shape (rows), run at every shape (columns), as % of that column's best")
    print(f"  {'winner at':>10s} {'configuration':40s}" + "".join(f"{'M=' + str(M):>10s}" for M in SHAPES))
    for Mw in SHAPES:
        x = best[Mw]
        cells = [f"{100 * t(best[M], M) / t(x, M):9.1f}%" if t(x, M) else f"{'-':>10s}" for M in SHAPES]
        print(f"  {'M=' + str(Mw):>10s} {x['id']:40s}" + "".join(cells))
    both = [x for x in recs if all(t(x, M) for M in SHAPES)]
    if both:
        worst = max(both, key=lambda x: min(t(best[M], M) / t(x, M) for M in SHAPES))
        print(f"  one configuration for all three shapes, best worst case: {worst['id']}: " +
              ", ".join(f"M={M} {100 * t(best[M], M) / t(worst, M):.1f}%" for M in SHAPES))
    fig, ax = plt.subplots(figsize=(5.4, 5))
    xs = [100 * t(best[128], 128) / t(x, 128) for x in both]
    ys = [100 * t(best[8192], 8192) / t(x, 8192) for x in both]
    ax.scatter(xs, ys, s=8, color="#177245", alpha=0.5)
    for M, mk in ((128, "s"), (1024, "^"), (8192, "D")):
        x = best[M]
        if x in both:
            ax.scatter([100 * t(best[128], 128) / t(x, 128)], [100 * t(best[8192], 8192) / t(x, 8192)], s=60,
                       marker=mk, color=COLORS[M], label=f"winner at M = {M:,}", zorder=3)
    ax.set_xlabel("at M = 128: % of the best there")
    ax.set_ylabel("at M = 8,192: % of the best there")
    ax.set_title("Problem 12: every configuration, two shapes")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    stamp(fig, r["facts"]["app_id"])
    fig.tight_layout()
    fig.savefig(out / "tune.png", dpi=160)


def final(folder):
    r = load(folder, "final")
    if not r:
        return
    print(f"\nFinal, run {r['facts']['app_id']}")
    shares = []
    for M, x in zip(SHAPES, r["records"]):
        res = x.get("results", {}).get(str(M), {})
        if "pct_cublas" in res:
            shares.append(res["pct_cublas"])
            print(f"  M={M:5d} {x['id']:40s} {res['time_us']:10.1f} us  {res['pct_cublas']:5.1f}% of cuBLAS")
    if len(shares) == 3:
        print(f"  geometric mean of the three shares: {math.prod(shares) ** (1 / 3):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    a = ap.parse_args()
    folder = Path(a.results)
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    ladder(folder, out)
    all_sweeps(folder, out)
    tune(folder, out)
    final(folder)
    print(f"\nfigures in {out}")
