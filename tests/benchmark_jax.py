"""Helpers to run the benchmarks with JAX on the CPU and on the GPU"""
import time

try:
    import jax
    import jax.numpy as jnp
    # Use float64 as for the NumPy benchmarks
    jax.config.update('jax_enable_x64', True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def jax_devices():
    """Get the devices to run the JAX benchmarks on

    Returns:
        List of tuples of name and device for the CPU and for
        the first GPU, if JAX is installed with GPU support
    """
    if not JAX_AVAILABLE:
        print("JAX is not installed, skipping the JAX benchmarks")
        return []

    devices = [("CPU", jax.devices("cpu")[0])]
    try:
        devices.append(("GPU", jax.devices("gpu")[0]))
    except RuntimeError:
        print("JAX found no GPU, running the JAX benchmarks on the CPU only")
    return devices


def to_device(array, device):
    """Copy a NumPy array to a JAX device"""
    return jax.device_put(jnp.asarray(array), device)


def measure(func, *args, repeats=3):
    """Measure the run time of a function returning JAX arrays

    Args:
        func: Function to measure
        *args: Arguments of the function
        repeats: Number of calls following the first call

    Returns:
        Tuple of the run time of the first call in seconds, which includes
        compilation, the shortest run time of the following calls in
        seconds and the result of the function
    """
    times = []
    for _ in range(repeats + 1):
        t0 = time.perf_counter()
        result = jax.block_until_ready(func(*args))
        times.append(time.perf_counter() - t0)
    return times[0], min(times[1:]), result
