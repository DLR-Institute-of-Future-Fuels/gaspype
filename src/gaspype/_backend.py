from typing import Any, Callable
from types import ModuleType
import importlib

_namespace_names = {'numpy': 'numpy', 'jax': 'jax.numpy'}


def get_backend(backend: str | ModuleType = 'numpy', device: Any = None) -> tuple[Any, Any]:
    """Get the array namespace and the device for a backend

    Args:
        backend: 'numpy', 'jax' or the module of a NumPy compatible
            array library (e.g. jax.numpy)
        device: Optional device for the JAX backend: A JAX device or
            a string like 'cpu', 'gpu' or 'gpu:1'

    Returns:
        Tuple of array namespace and device
    """
    if isinstance(backend, str):
        namespace: Any = importlib.import_module(_namespace_names.get(backend, backend))
    else:
        namespace = backend

    if device is not None:
        if not is_jax(namespace):
            raise ValueError('A device can only be selected for the JAX backend')
        if isinstance(device, str):
            import jax
            kind, _, index = device.partition(':')
            device = jax.devices(kind)[int(index or 0)]

    return namespace, device


def is_jax(namespace: Any) -> bool:
    """Check if the array namespace is jax.numpy"""
    return bool(namespace.__name__ == 'jax.numpy')


def as_float_array(a: Any, namespace: Any, device: Any = None) -> Any:
    """Convert a scalar, list or array to a floating point array

    Args:
        a: Scalar, list or array
        namespace: Array namespace of the returned array
        device: JAX device of the returned array or None for
            keeping the device or using the default device

    Returns:
        Array of the namespace
    """
    arr = namespace.asarray(a)
    if device is not None:
        import jax
        arr = jax.device_put(arr, device)
    return arr if arr.dtype.kind == 'f' else arr.astype(float)


def iterate(step: Callable[[Any], tuple[Any, Any]], x: Any, max_iterations: int, namespace: Any) -> Any:
    """Repeat an iteration step until it reports convergence

    Args:
        step: Function that takes the current values and returns a tuple
            of the new values and a boolean scalar array that is true if
            the iteration is converged
        x: Start values
        max_iterations: Maximum number of calls of step
        namespace: Array namespace of x

    Returns:
        Values returned by the last call of step
    """
    if is_jax(namespace):
        return _iterate_jax(step, x, max_iterations)

    for _ in range(max_iterations):
        x, converged = step(x)
        if converged:
            break
    return x


def _iterate_jax(step: Callable[[Any], tuple[Any, Any]], x: Any, max_iterations: int) -> Any:
    """Variant of iterate for JAX, which runs the iteration by
    jax.lax.while_loop if the values are traced (e.g. by jax.jit or jax.vmap)"""
    import jax

    x, converged = step(x)
    try:
        done = bool(converged)
    except jax.errors.TracerBoolConversionError:
        # The values are traced, therefore Python control flow can not depend on them
        def cond(state: tuple[Any, Any, Any]) -> Any:
            i, _, converged = state
            return (i < max_iterations) & ~converged

        def body(state: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            i, x, _ = state
            x, converged = step(x)
            return i + 1, x, converged

        return jax.lax.while_loop(cond, body, (1, x, converged))[1]

    for _ in range(max_iterations - 1):
        if done:
            break
        x, converged = step(x)
        done = bool(converged)
    return x
