# Assignment 2: climbing the matmul ladder, and tuning it for three shapes

This folder holds the GPU part of Assignment 2. The handout, `A2_Matmul_Ladder_and_Autotuning.pdf`, has the problems;
the programs here check and time your kernels on an H100 that you rent from Modal, run the one-knob sweeps and the
autotuner, and draw your figures.

The product is Qwen3-8B's MLP up-projection, `Y = X W`, with `W` of 4,096 x 12,288, in FP32 on the CUDA cores, for
128, 1,024 and 8,192 tokens. You write kernels 2 to 7 of the ladder from Lectures 7 and 9; kernel 1 is given.

| File | What it does |
|---|---|
| `kernels/` | One CUDA C file for each rung. `k1_naive.cu` is complete; the other six are yours to write. |
| `tune.py` | Your autotuner's search space and rules (Problem 12). |
| `best.json` | Your champion for each shape (Problem 13). It starts with the lectures' kernel 7. |
| `a2.py` | The harness: compiles, checks against FP64, times, and saves `results/<name>.json`. Do not change how it checks or times. |
| `plot.py` | Draws your figures and prints the tables, the regret matrix among them, from `results/`. |
| `my_element.py` | Prints your own element of `Y` from your email. Runs on your laptop. |
| `requirements.txt` | The Modal client and matplotlib, for your laptop. |
| `image-requirements.txt` | Every package the GPU machine installs, pinned by version and hash; the same image as Assignment 1. |

## 1. Get the code and log in

```bash
cd Kernel-Engg
git pull
cd assignment2
pip install -r requirements.txt
```

If you have not used Modal on this computer, run `modal setup` once. Modal keeps your key in `~/.modal.toml`; never
paste it into code, a chat, your report or your video.

## 2. Find your element

```bash
python my_element.py you@example.com
```

Use the email you registered for the workshop with. Nothing is sent anywhere.

## 3. Write a kernel, then check it

```bash
modal run a2.py::check
```

It runs every kernel in `kernels/` once with the lectures' sizes at all three shapes and prints `right` or `WRONG` for
each. A file that is still a `TODO` simply comes out wrong. If a kernel reads outside the matrices, the GPU context is
lost; the harness then names the configuration that did it and stops.

## 4. Measure

```bash
modal run a2.py::ladder              # Problem 6
modal run a2.py::sweep --knob tm     # Problem 8 (also: patch, tile, bk, warp, or all)
modal run a2.py::tune                # Problem 12: everything tune.py allows
modal run a2.py::final               # Problem 13: the configurations in best.json
python plot.py                       # figures/ and the tables
```

Every run prints its app ID (`ap-...`), the one [modal.com/apps](https://modal.com/apps) shows, and saves
`results/<name>.json`. A second run of the same command keeps the first as `<name>_run2.json`, so nothing is lost.

## What it costs

An H100 costs about $3.95 an hour on Modal, billed by the second. `check` takes a few seconds of GPU, `ladder` about
20 s, `sweep --knob all` about a minute, and `tune` about 1.3 s for each configuration (all of them are compiled at
once on the machine's CPU cores first; checking and timing at three shapes takes the rest). A tune of 600
configurations takes about 15 minutes, around $1. `tune` refuses to start with more than 600 unless you pass `--limit`.

## If something goes wrong

- **`modal: command not found`**: try `python3 -m pip install -r requirements.txt`, then `python3 -m modal run ...`.
- **`does not compile`**: the record in `results/*.json` holds the compiler's message.
- **`WRONG` at one shape only**: usually `M`, `N` and `K` mixed up in an index; draw the tile (Part A) and follow one
  thread.
- **After a run**: check [modal.com/apps](https://modal.com/apps). Nothing should still be running. If you stop a run
  with Ctrl+C, stop its app there too.
