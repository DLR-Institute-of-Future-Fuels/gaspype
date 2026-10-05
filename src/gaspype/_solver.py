from typing import Literal, Any, TYPE_CHECKING
import numpy as np
from ._main import elements, fluid, fluid_system
from .typing import NDFloat, FloatArray
from .constants import p0, epsy

if TYPE_CHECKING:
    def minimize(*a: Any, **b: Any) -> dict[str, FloatArray]:
        ...
else:
    try:
        from scipy.optimize import minimize
    except ImportError:
        def minimize(*a, **b):
            raise ImportError('scipy is required for the "gibs minimization" solver')


def set_solver(solver: Literal['gibs minimization', 'system of equations']) -> None:
    """
    Select a solver for chemical equilibrium.

    Solvers:
        - **system of equations** (default): Finds the root for a system of
          equations covering a minimal set of equilibrium equations and elemental balance.
          The minimal set of equilibrium equations is derived by SVD calculating null space.

        - **gibs minimization**: Minimizes the total Gibbs Enthalpy while keeping
          the elemental composition constant using the SLSQP implementation of scipy

    Args:
        solver: Name of the solver
    """
    global _equilibrium_solver
    if solver == 'gibs minimization':
        _equilibrium_solver = equilibrium_gmin
    elif solver == 'system of equations':
        _equilibrium_solver = equilibrium_eq
    else:
        raise ValueError('Unknown solver')


def get_solver() -> Literal['gibs minimization', 'system of equations']:
    """Returns the selected solver name.

    Returns:
        Solver name
    """
    if _equilibrium_solver == equilibrium_gmin:
        return 'gibs minimization'
    else:
        assert _equilibrium_solver == equilibrium_eq
        return 'system of equations'


def equilibrium_gmin(fs: fluid_system, element_composition: FloatArray,
                     t: float | FloatArray, p: float | FloatArray) -> FloatArray:
    """Calculate the equilibrium composition of a fluid based on minimizing the Gibbs free energy

    Args:
        fs: Fluid system
        element_composition: Elemental composition with shape (..., n_elements)
        t: Temperature in Kelvin, broadcastable to the batch shape
        p: Pressure in Pascal, broadcastable to the batch shape

    Returns:
        Molar amounts of species with shape (..., n_species), where ... is the
        broadcast shape of element_composition[..., 0], t and p
    """
    def element_balance(n: FloatArray, fs: fluid_system, ref: FloatArray) -> FloatArray:
        return np.dot(n, fs.array_species_elements) - ref  # type: ignore

    def gibbs_rt(n: FloatArray, grt: FloatArray, p_rel: float):  # type: ignore
        # Calculate G/(R*T)
        return np.sum(n * (grt + np.log(p_rel * n / np.sum(n) + epsy)))

    element_composition, t, p = _broadcast_inputs(element_composition, t, p)
    bnds = [(0, None) for _ in fs.species]
    start_composition_array = np.ones_like(fs.species, dtype=float)

    sol = np.zeros(t.shape + (len(fs.species),), dtype=NDFloat)
    for index in np.ndindex(t.shape):
        cons: dict[str, Any] = {'type': 'eq', 'fun': element_balance, 'args': [fs, element_composition[index]]}
        grt = fs.get_species_g_rt(float(t[index]))
        p_rel = float(p[index]) / p0
        sol[index] = minimize(gibbs_rt, start_composition_array, args=(grt, p_rel), method='SLSQP',
                              bounds=bnds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})['x']

    return sol


def _broadcast_inputs(element_composition: FloatArray, t: float | FloatArray,
                      p: float | FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Broadcast element composition (..., n_elements), t and p to a common batch shape"""
    element_composition = np.asarray(element_composition, dtype=NDFloat)
    t_arr = np.asarray(t, dtype=NDFloat)
    p_arr = np.asarray(p, dtype=NDFloat)
    batch_shape = np.broadcast_shapes(element_composition.shape[:-1], t_arr.shape, p_arr.shape)
    return (np.broadcast_to(element_composition, batch_shape + element_composition.shape[-1:]),
            np.broadcast_to(t_arr, batch_shape),
            np.broadcast_to(p_arr, batch_shape))


def equilibrium_eq(fs: fluid_system, element_composition: FloatArray,
                   t: float | FloatArray, p: float | FloatArray) -> FloatArray:
    """Calculate the equilibrium composition of a fluid based on equilibrium equations

    All equilibria of the batch are solved simultaneously by a vectorized Newton
    iteration.

    Args:
        fs: Fluid system
        element_composition: Elemental composition with shape (..., n_elements)
        t: Temperature in Kelvin, broadcastable to the batch shape
        p: Pressure in Pascal, broadcastable to the batch shape

    Returns:
        Molar amounts of species with shape (..., n_species), where ... is the
        broadcast shape of element_composition[..., 0], t and p
    """
    element_composition, t, p = _broadcast_inputs(element_composition, t, p)

    a = fs.array_stoichiometric_coefficients  # (n_reactions, n_species)
    species_elements = fs.array_species_elements  # (n_species, n_elements)
    a_sum = np.sum(a, axis=1)

    el_max = np.max(element_composition, axis=-1, keepdims=True)
    element_norm = element_composition / el_max
    element_norm_log = np.log(element_norm + epsy)

    # Log equilibrium constants for each reaction equation
    b = -fs.get_species_g_rt(t) @ a.T

    # Pressure corrected log equilibrium constants
    bp = b - a_sum * np.log(p / p0)[..., None]

    # Calculating the maximum possible amount for each species based on the elements
    species_max = np.min((element_norm[..., None, :] + epsy) / (species_elements + epsy), axis=-1)
    species_max_log = np.log(species_max + epsy)

    def residuals(logn: FloatArray) -> tuple[FloatArray, FloatArray]:
        n: FloatArray = np.exp(logn)  # n is the molar amount normalized by el_max
        n_sum = np.sum(n, axis=-1, keepdims=True)

        # Residuals from equilibrium equations:
        resid_eq = (logn - np.log(n_sum)) @ a.T - bp

        # Jacobian for equilibrium equations: a @ (I - 1 * n / n_sum)
        j_eq = a - a_sum[:, None] * (n / n_sum)[..., None, :]

        # Residuals from elemental balance:
        el_sum_norm = n @ species_elements + epsy
        resid_ab = np.log(el_sum_norm) - element_norm_log

        # Jacobian for elemental balance:
        j_ab = species_elements.T * n[..., None, :] / el_sum_norm[..., :, None]

        return (np.concatenate([resid_eq, resid_ab], axis=-1), np.concatenate([j_eq, j_ab], axis=-2))

    logn: FloatArray = species_max_log  # Set start values

    for _ in range(30):
        rF, J = residuals(logn)

        delta = np.linalg.solve(J, -rF[..., None])[..., 0]

        logn = np.minimum(logn + delta, species_max_log + 1)

        # Iterate until all equilibria of the batch are converged
        if np.all(np.linalg.norm(rF, axis=-1) < 1e-10):
            break

    n_eq: FloatArray = np.exp(logn) * el_max
    return n_eq


def equilibrium(f: fluid | elements, t: float | FloatArray, p: float | FloatArray = 1e5) -> fluid:
    """Calculate the isobaric equilibrium composition of a fluid at a given temperature and pressure

    The shapes of f, t and p are broadcast against each other, so
    multiple equilibria are solved in parallel.

    Args:
        f: Fluid or elements object
        t: Temperature in Kelvin
        p: Pressure in Pascal

    Returns:
        A new fluid object with the equilibrium composition
    """
    assert isinstance(f, (fluid, elements)), 'Argument f must be a fluid or elements'
    m_shape: int = f.fs.array_stoichiometric_coefficients.shape[0]
    if isinstance(f, fluid):
        if not m_shape:
            return f
    else:
        if not m_shape:
            def linalg_lstsq(array_elemental_composition: FloatArray, matrix: FloatArray) -> Any:
                # TODO: np.dot(np.linalg.pinv(a), b) is eqivalent to lstsq(a, b).
                # the constant np.linalg.pinv(a) can be precomputed for each fs.
                return np.dot(np.linalg.pinv(matrix), array_elemental_composition)

            # print('-->', f.array_elemental_composition.shape, f.fs.array_species_elements.transpose().shape)
            composition = np.apply_along_axis(linalg_lstsq, -1, f.array_elemental_composition, f.fs.array_species_elements.transpose())
            return fluid(composition, f.fs)

    assert np.min(f.array_elemental_composition) >= 0, 'Input element fractions must be 0 or positive'
    composition = _equilibrium_solver(f.fs, f.array_elemental_composition, t, p)
    return fluid(composition, f.fs)


_equilibrium_solver = equilibrium_eq
