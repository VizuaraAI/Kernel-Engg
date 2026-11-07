from typing import Any
import triton
import triton.language as tl
import torch
import torch.nn as nn

@triton.jit
def dropout_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.compile(fullgraph=True)
def dropout(x, p):
    output = torch.zeros_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    dropout_kernel[grid](x, output, n_elements, p, seed=10, BLOCK_SIZE=4)
    return output

class TritonDropoutLayer(nn.Module):
    def __init__(self, p: float = 0.5, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, x):
        return dropout(x, self.p)

batch_size = 32
dims = 128
device = torch.device("cuda:0")
batch_x = torch.ones(size=(batch_size, dims), dtype=torch.float32, device=device)
dropout_layer = TritonDropoutLayer(p=0.3)
batch_outputs = dropout_layer(batch_x)
print(batch_outputs.shape)
print(batch_outputs[0])

