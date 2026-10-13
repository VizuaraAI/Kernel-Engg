# Lecture 3: The dog on a real H100

In the lecture we followed one photo of a dog, 1024 x 768 pixels, through an H100: one thread for each pixel, blocks
of 16 x 16 threads, and warps of 32 threads that run each instruction together. The second hour showed what the blog's
kernels cost on a real H100, and how many warps disagree at an if. This folder runs the same kernels on a GPU you rent
from Modal, so you can see those numbers on your own screen.

| File | What it does |
|---|---|
| `dog.py` | Compiles the kernels on an H100, runs them on the dog and times them. It checks the gray, checkerboard and portrait images against the blog's, pixel by pixel. |
| `dog/` | The inputs: the dog, the mask that marks which pixels are dog, and in `dog/blog/` the blog's own images. |
| `requirements.txt` | The Modal client version we tested with. |

The kernels come from Vizuara's blog post
[The CUDA Programming Model](https://vizuaraai.github.io/cuda-programming-model/), written in CUDA C as close to the
blog's code as we could. The grayscale kernel is the one from section 4; the checkerboard and portrait mode come from
section 10, "Warp divergence, measured on the dog". The lecture's slides are in the workshop's course on Vizuara.

## What it costs

An H100 costs about $3.95 an hour on Modal, billed by the second. `modal run` keeps the GPU only while the program
runs: it starts a container on an H100, runs the kernels, sends the results back to your laptop and stops the
container, so one run costs a few cents. If something hangs on the GPU machine, the program stops its own container
after five minutes. New Modal accounts get free monthly credits; check [modal.com/pricing](https://modal.com/pricing)
for the current amounts.

## 1. Install Modal and log in

```bash
pip install modal
modal setup
```

`modal setup` opens your browser once. Modal then keeps your key on your computer, so you never paste a key into code,
a chat or a notebook. If you set up Modal for Lecture 1, you can skip this step.

## 2. Get the code

```bash
git clone https://github.com/VizuaraAI/Kernel-Engg
cd Kernel-Engg/lecture3
```

If you cloned the repository for Lecture 1, run `git pull` inside `Kernel-Engg` instead, then `cd lecture3`. The
folder's `requirements.txt` names the Modal client version we tested with, and `pip install -r requirements.txt`
installs exactly that version.

## 3. Run the dog

```bash
modal run dog.py
```

This rents one H100, compiles the kernels on it with NVRTC, launches them from Python with CuPy and times each one
with CUDA events: the median of many launches after a warm-up. The first run also builds the software image, which
takes a few minutes; later runs reuse it and finish in well under a minute.

While it runs, the GPU machine prints a line for each kernel it measures. Then your laptop prints a summary. Our test
run on 13 October 2026 printed:

```
NVIDIA H100 80GB HBM3 (132 SMs), measured 2026-10-13 09:47:02 UTC

what the dog costs, one dog per launch:
  kernel A: grayscale, one dog           4.2 µs
  portrait mode                          4.3 µs
  checkerboard                           4.9 µs
  an empty kernel                        1.8 µs   (no kernel can take less)

pixel (579, 329): RGB [48, 34, 19] -> gray 36 (the blog: 36)

checks against the blog's own images:
  gray               786,411 of 786,432 pixels identical to dog/blog/gray.png (largest difference: 1)
  gray_without_fma   786,432 of 786,432 pixels identical to dog/blog/gray.png, compiled with --fmad=false
  checker            786,432 of 786,432 pixels identical to dog/blog/checker.png
  portrait           786,432 of 786,432 pixels identical to dog/blog/portrait.png

divergence: warps whose 32 threads disagree, counted on the GPU with __ballot_sync, of 24,576:
  checkerboard                           24,576
  portrait mode                             659
  top half (y < 384)                          0

divergence: heavy branches, 512 steps of arithmetic a side:
  checkerboard                         109.3 µs
  portrait mode                         63.2 µs
  top half (y < 384)                    56.7 µs
  portrait condition, all-dog mask      57.1 µs
```

## 4. Match the numbers to the slides

The two "Measured" slides in the lecture's second hour show our run of 1 October 2026. This is where each of their
numbers appears in the printout, next to what our test run printed:

| Slide | On the slide (1 October) | In the printout | Our test run (13 October) |
|---|---|---|---|
| What the dog costs | kernel A, grayscale: 4.2 µs | `kernel A: grayscale, one dog` | 4.2 µs |
| What the dog costs | portrait mode: 4.4 µs | `portrait mode` | 4.3 µs |
| What the dog costs | checkerboard: 5.0 µs | `checkerboard` | 4.9 µs |
| What the dog costs | an empty launch: 2.0 µs | `an empty kernel` | 1.8 µs |
| Divergence | warps that disagree: checkerboard 24,576 of 24,576, portrait mode 659, top half 0 | the first `divergence` block | 24,576, 659 and 0 |
| Divergence | light branches: checkerboard 5.0 µs against portrait mode 4.4 µs | `checkerboard` and `portrait mode` under `what the dog costs` | 4.9 and 4.3 µs |
| Divergence | heavy branches: checkerboard 108.5 µs against top half 56.3 µs, 1.9 times as long | the second `divergence` block | 109.3 and 56.7 µs, 1.9 times as long |

The heavy branches have two more lines that are not on the slide. Portrait mode took 62.7 µs on 1 October (63.2 µs in
our test run), a little longer than the top half, because 659 of its warps run both sides. The last line uses portrait
mode's condition with a mask that is dog everywhere: every thread still reads its byte of the mask, but no warp
disagrees. It took 56.7 µs on 1 October (57.1 µs in our test run), about the same as the top half. So the extra time
in portrait mode comes from the warps that disagree, not from reading the mask.

The image checks compare each picture the GPU made with the blog's own. The checkerboard and portrait mode match in
all 786,432 pixels. The gray dog differs in 21 pixels, by 1 each. The GPU computes 0.299 R + 0.587 G + 0.114 B with
fused multiply-adds, which round once where a separate multiply and add round twice, and for those 21 pixels the
difference in the last bit carries the result across a whole number. The next check compiles the same kernel with
`--fmad=false`, which turns the fused multiply-adds off, and then all 786,432 pixels match. Pixel (579, 329), on the
rim of the dog's right eye, is the one the lecture followed through every film. It comes out as 36, as in the blog.

## Your numbers

Your times will differ a little from ours, and from one run to the next. The two runs in the table, twelve days apart,
show how much: every time but one moved by 2 percent or less, and the empty kernel went from 2.0 to 1.8 µs. The counts
and the image checks do not move, because they depend only on the dog, the mask and the code. Every run should give
24,576, 659 and 0 warps, pixel (579, 329) at 36, and the same four image checks.

## Where the results land

The run writes two folders next to `dog.py`, and the last line of its printout names them:

- `out/` holds the three images the GPU made: `gray.png`, `checker.png` and `portrait.png`. Open them next to the
  blog's images in `dog/blog/`.
- `data/dog_kernels.json` holds every number from the run, with the GPU and the time of the measurement.

Both folders are listed in the repository's `.gitignore`, so git leaves them out.

## If something goes wrong

- **`modal: command not found`**: the install from step 1 went to a different Python than your shell uses. Try
  `python3 -m pip install modal`, then `python3 -m modal setup` and `python3 -m modal run dog.py`.
- **`git clone` says `Kernel-Engg` already exists**: you cloned the repository for Lecture 1. Run
  `git -C Kernel-Engg pull`, then `cd Kernel-Engg/lecture3`.
