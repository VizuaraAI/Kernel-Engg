# Kernel Engineering

Code for the Kernel Engineering Workshop by [Vizuara](https://vizuara.ai). Each lecture has its own folder with the
programs we run in class, so you can repeat the measurements from the lecture on a GPU of your own.

The GPU work runs on [Modal](https://modal.com), a cloud service that rents you a GPU by the second. Your laptop only
sends the code and shows the results, so you need a Modal account and Python 3 on your laptop (we tested with Python
3.11), not a GPU. New Modal accounts get free monthly credits; see [modal.com/pricing](https://modal.com/pricing).

| Folder | Lecture | What you run |
|---|---|---|
| [`lecture1`](lecture1) | Lecture 1: How fast can this go? | Measure an H100's two speed limits, serve Qwen3-8B on it with vLLM, and chat with your model from a browser. |
| [`lecture3`](lecture3) | Lecture 3: How does a GPU run your code? | Time the lecture's CUDA kernels on a photo of a dog on one H100, and count the warps whose 32 threads disagree at an if. |
| [`assignment1`](assignment1) | Assignment 1: Rooflines, arithmetic intensity and the memory hierarchy | The handout and its four experiments on one H100: the GPU's two ceilings and its L2, the roofline traced with real matmuls, coalescing and bytes in flight, and RoPE four ways. |

We add a folder here after each lecture is taught.

## Nvidia GPU Whitepapers

* Blackwell Architecture: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
* Hopper Architecture: https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c
* Ada Architecture: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
* Ampere Architecture: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
* Turing Architecture: https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
* Tesla Architecture: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf