import cuda.tile as ct
import torch

# Sizes of the tiles used along
# M, K and N dimensions
TILE_K = 32
TILE_M = 32
TILE_N = 32

@ct.kernel
def matmul_kernel(
    x_ptr: ct.IndexedWithInt64,
    y_ptr: ct.IndexedWithInt64,
    prod_ptr: ct.IndexedWithInt64,
    TILE_M: ct.Constant[int], TILE_K: ct.Constant[int], TILE_N: ct.Constant[int]
):
    bid_m = ct.bid(axis=0)
    bid_n = ct.bid(axis=1)
    num_tiles_along_k = ct.num_tiles(x_ptr, axis=1, shape=(TILE_M, TILE_K))
    # or num_tiles_along_k = ct.num_tiles(y_ptr, axis=0, shape=(TILE_K, TILE_N))
    accumulator = ct.zeros(shape=(TILE_M, TILE_N), dtype=ct.float32)
    for t in range(num_tiles_along_k):
        # Note: `index` belongs to the tile-space, see documentation of
        # tl.load and tl.store functions.
        tile_x = ct.load(
            x_ptr,
            index=(bid_m, t),
            shape=(TILE_M, TILE_K)
        )
        tile_y = ct.load(
            y_ptr,
            index=(t, bid_n),
            shape=(TILE_K, TILE_N)
        )
        accumulator += ct.matmul(tile_x, tile_y)
    ct.store(
        prod_ptr,
        index=(bid_m, bid_n),
        tile=accumulator
    )

def matmul(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    M, K = x.shape
    K, N = y.shape
    product = torch.empty(size=(M, N), dtype=x.dtype, device=x.device)
    grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        matmul_kernel,
        (x, y, product, TILE_M, TILE_K, TILE_N)
    )
    return product

def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    M = 1024
    K = 1024
    N = 1024
    x = torch.rand(size=(M, K), dtype=torch.float32, device=device)
    y = torch.rand(size=(K, N), dtype=torch.float32, device=device)
    product = matmul(x, y)

    torch.testing.assert_close(product, x @ y)

main()