"""Lecture 1 practical: serve Qwen3-8B on one H100 with vLLM, on your own Modal account.

    pip install modal
    modal setup                        # log in once in the browser; Modal keeps your key on this machine
    modal run serve.py::gpu_info       # ask the GPU for its configuration and measure its two speed limits
    modal serve serve.py               # start the model and print its URL; Ctrl+C stops the GPU

`modal serve` keeps the GPU only while this command runs, so closing the terminal is enough to stop the bill
(an H100 costs about $3.95 an hour on Modal). The first start takes a few minutes: the image is built once, and the
16 GB of weights are downloaded and loaded onto the GPU.
"""
import subprocess
import time

import modal

MODEL = "Qwen/Qwen3-8B"
app = modal.App("my-first-llm")
image = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .pip_install("vllm==0.30.0", "huggingface-hub==1.33.0")
         .env({"CUDA_HOME": "/usr/local/cuda", "VLLM_USE_FLASHINFER_SAMPLER": "0"}))


@app.function(gpu="H100", image=image)
def gpu_info():
    import torch

    p = torch.cuda.get_device_properties(0)
    print(f"GPU              {p.name}")
    print(f"SMs              {p.multi_processor_count}  ({4 * p.multi_processor_count} tensor cores, 4 per SM)")
    print(f"HBM memory       {p.total_memory / 2**30:.0f} GiB")
    print(f"L2 cache         {getattr(p, 'L2_cache_size', 0) / 2**20:.0f} MiB")

    def seconds(fn, reps=20):                  # average time of fn on the GPU
        fn(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps

    x = torch.ones(2**30, dtype=torch.bfloat16, device="cuda")          # 2 GiB of numbers
    print(f"memory speed     {x.numel() * 2 / seconds(lambda: x.sum()) / 1e12:.2f} TB/s    (datasheet: 3.35)")
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    print(f"math speed       {2 * 8192**3 / seconds(lambda: a @ a) / 1e12:.0f} TFLOP/s  (datasheet: 989)")
    subprocess.run(["nvidia-smi"])


@app.function(gpu="H100", image=image, max_containers=1, scaledown_window=300, timeout=3600)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    subprocess.Popen(["vllm", "serve", MODEL, "--served-model-name", "qwen3-8b", "--max-model-len", "4096",
                      "--host", "0.0.0.0", "--port", "8000"])
