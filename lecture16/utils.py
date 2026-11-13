from triton.testing import do_bench
import torch

def test_kernel(kernel_fn, spec_fn, *args, rtol=None, atol=None):
    """Test a Helion kernel against a reference implementation."""
    # Run our implementation
    result = kernel_fn(*args)
    # Run reference implementation
    expected = spec_fn(*args)

    # Check if results match
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
    print("✅ Results Match ✅")

def benchmark_kernel(kernel_fn, *args, **kwargs):
    """Benchmark a Helion kernel."""
    no_args = lambda: kernel_fn(*args, **kwargs)
    time_in_ms = do_bench(no_args)
    print(f"⏱ Time: {time_in_ms} ms")

def compare_implementations(kernel_fn, spec_fn, *args, **kwargs):
    """Benchmark a Helion kernel and its reference implementation."""
    kernel_no_args = lambda: kernel_fn(*args, **kwargs)
    spec_no_args = lambda: spec_fn(*args, **kwargs)
    kernel_time = do_bench(kernel_no_args)
    spec_time = do_bench(spec_no_args)
    print(f"⏱ Helion Kernel Time: {kernel_time:.3f} ms, PyTorch Reference Time: {spec_time:.3f} ms, Speedup: {spec_time/kernel_time:.3f}x")