from . import R, F, oxygen_partial_pressure, fluid, elements, FloatArray
import numpy as np

# Each O2 molecule is transported as two ions and each ion has a charged of 2e
z_O2 = 4


def voltage_potential(f1: fluid | elements, f2: fluid | elements, t: float, p: float = 1e5) -> FloatArray:
    p1 = oxygen_partial_pressure(f1, t, p)
    p2 = oxygen_partial_pressure(f2, t, p)
    return R * t / (z_O2 * F) * np.log(p2 / p1)
