import helion
import helion.language as hl
import torch

@helion.kernel(autotune_effort="none")
def layer_norm_kernel(x: torch.Tensor):
    m, n = x.shape

    output = torch.empty_like(x)
    for tile_m in hl.tile(m):

        # compute mean
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            x_tile = x[tile_m, tile_n]
            acc += x_tile.sum(dim=1)
        mean = acc / n

        # compute standard deviation
        var_acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            x_tile = x[tile_m, tile_n]
            x_tile -= mean
            var_acc += torch.pow(x_tile, 2.0).sum(dim=1)
        std = torch.sqrt(var_acc / (n + 1e-7))

        # normalize
        for tile_n in hl.tile(n):
            x_tile = x[tile_m, tile_n]
            normalized = (x_tile - mean[:, None]) / std[:, None]
            output[tile_m, tile_n] = normalized

    return output

def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    x = torch.rand(size=(2,3), dtype=torch.float32, device=device)
    torch_layer_norm = torch.nn.LayerNorm(normalized_shape=3, device=device)
    y_torch = torch_layer_norm(x)
    y = layer_norm_kernel(x)
    torch.testing.assert_close(y, y_torch, atol=1e-3, rtol=1e-3)

main()
