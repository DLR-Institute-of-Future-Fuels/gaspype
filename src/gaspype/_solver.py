from typing import Literal, Any, TYPE_CHECKING
import numpy as np
from ._main import elements, fluid, fluid_system
from .typing import NDFloat, FloatArray
from .constants import p0, epsy

if TYPE_CHECKING:
    from scipy.optimize import minimize
else:
    try:
        from scipy.optimize import minimize
    except ImportError:
        def minimize(*a: Any, **b: Any) -> Any:
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


def equilibrium_gmin(fs: fluid_system, element_composition: FloatArray, t: float, p: float) -> FloatArray:
    """Calculate the equilibrium composition of a fluid based on minimizing the Gibbs free energy"""
    def element_balance(n: FloatArray, fs: fluid_system, ref: FloatArray) -> FloatArray:
        return np.dot(n, fs.array_species_elements) - ref  # type: ignore

    def gibbs_rt(n: FloatArray, grt: FloatArray, p_rel: float):  # type: ignore
        # Calculate G/(R*T)
        return np.sum(n * (grt + np.log(p_rel * n / np.sum(n) + epsy)))

    cons: dict[str, Any] = {'type': 'eq', 'fun': element_balance, 'args': [fs, element_composition]}
    bnds = [(0, None) for _ in fs.species]
    grt = fs.get_species_g_rt(t)
    p_rel = p / p0

    start_composition_array = np.ones_like(fs.species, dtype=float)
    sol = np.array(minimize(gibbs_rt, start_composition_array, args=(grt, p_rel), method='SLSQP',
                   bounds=bnds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})['x'], dtype=NDFloat)  # type: ignore

    return sol


def equilibrium_eq(fs: fluid_system, element_composition: FloatArray, t: float, p: float) -> FloatArray:
    """Calculate the equilibrium composition of a fluid based on equilibrium equations"""
    el_max = np.max(element_composition)
    element_norm = element_composition / el_max
    element_norm_log = np.log(element_norm + epsy)

    a = fs.array_stoichiometric_coefficients
    el_matrix = fs.array_species_elements.T

    # Log equilibrium constants for each reaction equation
    b = -np.sum(fs.get_species_g_rt(t) * a, axis=1)

    # Pressure corrected log equilibrium constants
    bp = b - np.sum(a * np.log(p / p0), axis=1)

    # Calculating the maximum possible amount for each species based on the elements
    species_max = np.min((element_norm + epsy) / (fs.array_species_elements + epsy), axis=1)
    species_max_log = np.log(species_max + epsy)

    # Prepare constant arrays
    j_eq_eye = np.eye(len(species_max))
    j_eq_ones = np.ones((len(species_max), 1))

    def residuals(logn: FloatArray) -> tuple[FloatArray, FloatArray]:
        n: FloatArray = np.exp(logn)  # n is the molar amount normalized by el_max
        n_sum = np.sum(n)

        # Residuals from equilibrium equations:
        resid_eq = a @ (logn - np.log(n_sum)) - bp

        # Jacobian for equilibrium equations:
        j_eq = a @ (j_eq_eye - j_eq_ones * n / np.sum(n))

        # Residuals from elemental balance:
        el_sum_norm = np.dot(el_matrix, n) + epsy
        resid_ab = np.log(el_sum_norm) - element_norm_log
        #print(f'* resid_eq: {resid_eq}      resid_ab: {resid_ab}            {element_norm}')

        # Jacobian for elemental balance:
        j_ab = el_matrix * n / el_sum_norm[:, None]

        return (np.hstack([resid_eq, resid_ab]), np.concatenate([j_eq, j_ab], axis=0))

    logn: FloatArray = species_max_log  # Set start values

    for i in range(30):
        rF, J = residuals(logn)

        delta = np.linalg.solve(J, -rF)

        logn = logn + delta
        logn = np.minimum(logn, species_max_log + 1)

        #print(f'{i} F: {np.linalg.norm(rF):.5f} lognmin={np.min(logn):.3f}, lognmax={np.max(logn):.3f}, delta={np.linalg.norm(delta):.3f} logn=')
        if np.linalg.norm(rF) < 1e-10:
            #print(f'Converged in {i} iterations')
            break

    n = np.exp(logn)

    return n * el_max


def equilibrium(f: fluid | elements, t: float | FloatArray, p: float = 1e5) -> fluid:
    """Calculate the isobaric equilibrium composition of a fluid at a given temperature and pressure"

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
    if isinstance(t, np.ndarray):
        assert f.shape == tuple(), 'Multidimensional temperature can currently only used for 0D fluids'
        t_composition = np.zeros(t.shape + (f.fs.array_species_elements.shape[0],))
        for t_index in np.ndindex(t.shape):
            t_composition[t_index] = _equilibrium_solver(f.fs, f.array_elemental_composition, float(t[t_index]), p)
        return fluid(t_composition, f.fs)
    else:
        composition = np.ones(f.shape + (len(f.fs.species),), dtype=float)
        for index in np.ndindex(f.shape):
            # print(composition.shape, index, _equilibrium(f.fs, f._element_composition[index], t, p))
            composition[index] = _equilibrium_solver(f.fs, f.array_elemental_composition[index], t, p)
        return fluid(composition, f.fs)


_equilibrium_solver = equilibrium_eq
