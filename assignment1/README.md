# Assignment 1: rooflines, arithmetic intensity and the memory hierarchy

This folder holds the GPU part of Assignment 1. The handout, `A1_Rooflines_and_Memory.pdf`, has the problems; the
programs here run its four experiments (Problems 7 to 10) on an H100 that you rent from Modal, and draw your figures.
You do not write a kernel in this assignment. You read the code, change the settings marked `TUNE`, run it, and explain
what you measure.

| File | What it does |
|---|---|
| `a1.py` | The four experiments. Each one rents an H100, measures, prints a summary and saves `results/<name>.json`. |
| `plot.py` | Draws the figures for your report from `results/`, on your laptop. Two functions in it are yours to complete. |
| `requirements.txt` | The Modal client and the matplotlib version we tested with, for your laptop. |
| `image-requirements.txt` | Every package the GPU machine installs, pinned by version and hash. You do not need to touch it. |

## What it costs

An H100 costs about $3.95 an hour on Modal, billed by the second. Each experiment keeps the GPU for under a minute
and stops it when it ends, so one pass through all four costs about 15 cents, and ten passes while you tune stay under
$2. New Modal accounts get free monthly credits; check [modal.com/pricing](https://modal.com/pricing)
for the current amounts. If something hangs on the GPU machine, each experiment stops its own container after 15 to 20
minutes.

## 1. Get the code

```bash
git clone https://github.com/VizuaraAI/Kernel-Engg
cd Kernel-Engg/assignment1
```

If you cloned the repository for a lecture, run `git pull` inside `Kernel-Engg` instead, then `cd assignment1`.

## 2. Install Modal and log in

```bash
pip install -r requirements.txt
modal setup
```

`modal setup` opens your browser once. Modal then keeps your key on your computer, in `~/.modal.toml`, so you never
paste a key into code, a chat, your report or your video. If you set up Modal for an earlier lecture, you can skip the
login.

## 3. Run the experiments

```bash
modal run a1.py::ceilings     # Problem 7: copy speed against buffer size; the matmul and FP32 peaks
modal run a1.py::roofline     # Problem 8: Y = X @ W for 1 to 8,192 tokens
modal run a1.py::memory       # Problem 9: strided reads; bytes in flight per thread
modal run a1.py::rope         # Problem 10: RoPE four ways, from 1 token to 131,072
```

The first run builds the software image, which takes two or three minutes; later runs reuse it. While a run works, the
GPU machine prints a line for each measurement; at the end your laptop prints a summary table and saves the numbers.

## 4. Draw your figures

```bash
python plot.py
```

It writes `figures/*.png` from whatever is in `results/`. Two figures need something from you first: complete
`arithmetic_intensity()` in `plot.py` with your answer to Problem 8(a), and `strided_prediction()` with your model from
Problem 9(a).

## Your numbers

Your numbers will be close to ours, not identical, and they move a little from one run to the next. That is part of
the assignment: Problem 7 asks how much they move. Each results file records the GPU, its clocks, the time and the
run's IDs; put the printed summaries, with their IDs, in your report's appendix. The app ID (`ap-...`) is the one your
dashboard at [modal.com/apps](https://modal.com/apps) shows. If you set `GPU` to another GPU for bonus B1, its results
are saved next to your H100 results, for example `results/ceilings_L4.json`, so nothing is overwritten.

## If something goes wrong

- **`modal: command not found`**: the install went to a different Python than your shell uses. Try
  `python3 -m pip install -r requirements.txt`, then `python3 -m modal setup` and `python3 -m modal run a1.py::ceilings`.
- **A run stops with an error about memory or a missing GPU**: check that `GPU = "H100"` near the top of `a1.py`.
  The sizes are chosen for an 80 GB H100.
- **The run seems stuck at the start**: the first run builds the image. Watch it at
  [modal.com/apps](https://modal.com/apps).
- **After a run**: check [modal.com/apps](https://modal.com/apps). Nothing should still be running.
