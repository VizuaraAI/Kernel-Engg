# Kernel Engineering

Code for the Kernel Engineering Workshop by [Vizuara](https://vizuara.ai). Each lecture has its own folder with the
programs we run in class, so you can repeat the measurements from the lecture on a GPU of your own.

The GPU work runs on [Modal](https://modal.com), a cloud service that rents you a GPU by the second. Your laptop only
sends the code and shows the results, so you need a Modal account and Python 3 on your laptop (we tested with Python
3.11), not a GPU. New Modal accounts get free monthly credits; see [modal.com/pricing](https://modal.com/pricing).

| Folder | Lecture | What you run |
|---|---|---|
| [`lecture1`](lecture1) | Lecture 1: How fast can this go? | Measure an H100's two speed limits, serve Qwen3-8B on it with vLLM, and chat with your model from a browser. |

We add a folder here after each lecture is taught.
