# Lecture 1: Your own LLM on one H100

In the lecture we measured how fast one H100 can make a chatbot write. This folder lets you serve the same model,
Qwen3-8B, with vLLM on a GPU you rent from Modal, talk to it from your laptop or phone, and see the lecture's numbers on
your own screen.

| File | What it does |
|---|---|
| `serve.py` | Measures the GPU (`gpu_info`) and serves the model (`serve`). |
| `chat.html` | A chat page that talks to your server and shows how fast each answer arrives. |
| `requirements.txt` | The Modal client version we tested with. |

The lecture's slides are at [kernel-cohort.vercel.app/lecture1/v3](https://kernel-cohort.vercel.app/lecture1/v3/),
and the same practical, step by step, is at
[kernel-cohort.vercel.app/lecture1/v3/practical](https://kernel-cohort.vercel.app/lecture1/v3/practical/).

## What it costs

An H100 costs about $3.95 an hour on Modal, billed by the second. `modal serve` keeps the GPU only while the command
runs: press Ctrl+C, or close the terminal, and it stops. A 20-minute session costs about $1.30. New Modal accounts get
free monthly credits; check [modal.com/pricing](https://modal.com/pricing) for the current amounts.

## 1. Install Modal and log in

```bash
pip install -r requirements.txt
modal setup
```

`modal setup` opens your browser once. Modal then keeps your key on your computer, so you never paste a key into code,
a chat or a notebook.

## 2. Get the code

```bash
git clone https://github.com/VizuaraAI/Kernel-Engg.git
cd Kernel-Engg/lecture1
```

Run step 1 from inside this folder, or install Modal with `pip install modal`.

## 3. Ask the GPU for its configuration

```bash
modal run serve.py::gpu_info
```

This rents an H100 for a few seconds, reports its parts and measures its two speed limits: how fast it reads its
memory, and how fast it multiplies matrices. The first run also builds the software image, which takes a few minutes;
later runs reuse it. Ours printed:

```
GPU              NVIDIA H100 80GB HBM3
SMs              132  (528 tensor cores, 4 per SM)
HBM memory       79 GiB
L2 cache         50 MiB
memory speed     3.06 TB/s    (datasheet: 3.35)
math speed       810 TFLOP/s  (datasheet: 989)
```

Your two speeds will differ a little from ours, and from one run to the next.

## 4. Start the model

```bash
modal serve serve.py
```

It prints a web address ending in `-dev.modal.run`. The first start takes about five minutes: Modal builds the image
once, then downloads the model's 16 GB of weights and loads them onto the GPU. You can watch it happen at
[modal.com/apps](https://modal.com/apps): your app, its H100 container and the logs.

## 5. Talk to it

Open `chat.html` in your browser, paste the address from step 4 and ask anything. The answer streams back, and the page
shows the speed. To chat from your phone, open the hosted copy of the same page,
[kernel-cohort.vercel.app/lecture1/v3/practical/chat.html](https://kernel-cohort.vercel.app/lecture1/v3/practical/chat.html),
and paste the same address.

On our H100 one answer ran at 147 tokens a second, 6.8 milliseconds per token. For every token, the GPU reads about
15 GB of weights from its memory: all of them except the embedding table, of which it needs only one row a token. That
reading, not the arithmetic, sets the speed of a single user.

## 6. Stop the GPU

Press Ctrl+C in the terminal where `modal serve` runs. Then check [modal.com/apps](https://modal.com/apps): nothing
should be running. If `my-first-llm` still shows as running after a minute, stop it there, or run
`modal app stop my-first-llm`.

While your server runs, anyone who has its address can send it questions and spend your GPU time. Stop it when you
finish, and do not post the address anywhere public.

## 7. From one user to many

One user leaves most of the GPU's arithmetic idle. The
[users simulator](https://kernel-cohort.vercel.app/lecture1/v3/sim/) replays what our H100 measured with 1 to 1,024
users at once: move the slider and watch the operations per byte, the speed of each user, and the share of the GPU's
math in use.

## If something goes wrong

- **`modal: command not found`**: the install from step 1 went to a different Python than your shell uses. Try
  `python3 -m pip install modal`, then `python3 -m modal setup` and `python3 -m modal serve serve.py`.
- **The first answer takes minutes**: the model is still loading. The chat page says so after three seconds; wait for
  the first start to finish (step 4).
- **The chat page cannot reach your server**: check that `modal serve serve.py` is still running, and that you pasted
  the whole address, starting with `https://`.
