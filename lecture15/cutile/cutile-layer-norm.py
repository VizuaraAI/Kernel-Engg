import cuda.tile as ct
import torch
import math

TILE_N = 1024


@ct.kernel
def layer_norm_kernel(
        x_ptr: ct.IndexedWithInt64,
        out_ptr: ct.IndexedWithInt64,
        N: ct.Constant[int],
        TILE_N: ct.Constant[int]
):
    bid_m = ct.bid(axis=0)
    num_tiles = ct.num_tiles(x_ptr, axis=1, shape=(1, TILE_N))

    # compute mean
    mean = ct.full((1, TILE_N), 0.0, dtype=ct.float32)
    for t in range(num_tiles):
        tile = ct.load(x_ptr, index=(bid_m, t), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        mean += tile
    mean = ct.sum(mean, axis=1) / N

    # compute variance
    variance = ct.full((1, TILE_N), 0.0, dtype=ct.float32)
    for t in range(num_tiles):
        tile = ct.load(x_ptr, index=(bid_m, t), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        mask = ((t * TILE_N) + ct.arange(TILE_N, dtype=ct.int32)) < N
        centered_tile = ct.where(mask, tile - mean, 0.0)
        variance += centered_tile**2
    variance = ct.sum(variance, axis=1) / N
    std = ct.sqrt(variance + 1e-10)

    # normalize
    for t in range(num_tiles):
        tile = ct.load(x_ptr, index=(bid_m, t), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        tile = (tile - mean) / std
        ct.store(out_ptr, index=(bid_m, t), tile=tile)

def layer_norm(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    output = torch.zeros_like(x)
    grid = (M, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        layer_norm_kernel,
        (x, output, N, TILE_N)
    )
    return output


def main():
    torch.manual_seed(0)

    device = torch.device("cuda:0")

    x = torch.rand(size=(2,3), dtype=torch.float32, device=device) * 10
    x = torch.floor(x)
    print(x)
    p = torch.nn.LayerNorm(normalized_shape=3, device=device)
    y = p(x)
    print(y)

    y1 = layer_norm(x)
    print(y1)


main()
