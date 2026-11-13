import cuda.tile as ct
import torch
import math

TILE = 1024

@ct.kernel
def vec_add_kernel(
    x_ptr: ct.IndexedWithInt64,
    y_ptr: ct.IndexedWithInt64,
    sum_ptr: ct.IndexedWithInt64,
    TILE: ct.Constant[int]
):
    bid = ct.bid(axis=0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,))
    y_tile = ct.load(y_ptr, index=(bid,), shape=(TILE,))
    sum_tile = x_tile + y_tile
    ct.store(sum_ptr, index=(bid,), tile=sum_tile)

def vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sum = torch.zeros_like(x)
    num_elements = sum.numel()
    grid = (math.ceil(num_elements / TILE), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        vec_add_kernel,
        (x, y, sum, TILE)
    )
    return sum

def main():
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    size = 1024 * 1024
    x = torch.rand(size=(size, ), dtype=torch.float32, device=device)
    y = torch.rand(size=(size, ), dtype=torch.float32, device=device)

    sum = vec_add(x, y)

    torch.testing.assert_close(sum, x + y)

    # export the TileIR bytecode
    array_pointer_constraint = ct.compilation.ArrayConstraint(
        dtype=ct.float32,
        ndim=1,
        index_dtype=ct.int64,
        stride_lower_bound_incl=0,
        alias_groups=[],
        may_alias_internally=False,
    )

    ct.compilation.export_kernel(
        kernel=vec_add_kernel,
        signatures=[
            ct.compilation.KernelSignature(
                parameters=[
                    array_pointer_constraint,
                    array_pointer_constraint,
                    array_pointer_constraint,
                    ct.compilation.ConstantConstraint(value=TILE),
                ],
                calling_convention=ct.compilation.CallingConvention.cutile_python_v1()
            )
        ],
        output_format="cubin",
        output_file="cutile-vec-add.cubin",
        gpu_code="sm_89"
    )

main()