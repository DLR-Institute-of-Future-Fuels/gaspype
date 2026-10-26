from typing import Literal, Any, TYPE_CHECKING
import numpy as np
from ._main import elements, fluid, fluid_system
from .typing import NDFloat, FloatArray
from .constants import p0, epsy, R
from ._backend import iterate

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

    assert fs.np is np, 'The gibs minimization solver requires a fluid system with the NumPy backend'
    element_composition, t, p = _broadcast_inputs(fs, element_composition, t, p)
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


def _broadcast_inputs(fs: fluid_system, element_composition: FloatArray, t: float | FloatArray,
                      p: float | FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Broadcast element composition (..., n_elements), t and p to a common batch shape"""
    np = fs.np
    element_composition = fs._asarray(element_composition)
    t_arr = fs._asarray(t)
    p_arr = fs._asarray(p)
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
    np = fs.np
    element_composition, t, p = _broadcast_inputs(fs, element_composition, t, p)

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

    # The residuals can not get smaller than the precision of the data type
    tol = max(1e-10, 1e3 * float(np.finfo(species_max_log.dtype).eps))

    def newton_step(logn: FloatArray) -> tuple[FloatArray, Any]:
        rF, J = residuals(logn)

        delta = np.linalg.solve(J, -rF[..., None])[..., 0]

        # Iterate until all equilibria of the batch are converged
        converged = np.all(np.linalg.norm(rF, axis=-1) < tol)

        return np.minimum(logn + delta, species_max_log + 1), converged

    # Start values are the maximum possible amounts of the species
    logn: FloatArray = iterate(newton_step, species_max_log, 30, np)

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
            # Least squares solution of: species_elements.T @ composition = elemental_composition
            # TODO: the constant pinv(species_elements) can be precomputed for each fs.
            np = f.fs.np
            composition = np.dot(f.array_elemental_composition, np.linalg.pinv(f.fs.array_species_elements))
            return fluid(composition, f.fs)

    composition = _equilibrium_solver(f.fs, f.array_elemental_composition, t, p)
    return fluid(composition, f.fs)


def _t_range(fs: fluid_system) -> tuple[float, float]:
    """Temperature range in Kelvin covered by the property tables of the fluid system"""
    return float(fs._t_offset), float(fs._t_offset + fs._h_array.shape[0] - 1)


def equilibrium_eq_hp(fs: fluid_system, element_composition: FloatArray, h: float | FloatArray,
                      p: float | FloatArray, t_guess: float | FloatArray) -> tuple[FloatArray, FloatArray]:
    """Calculate the equilibrium composition and temperature of a fluid at a given
    enthalpy and pressure based on equilibrium equations

    The equation system of equilibrium_eq is extended by the energy balance
    and the logarithm of the temperature as additional unknown. All equilibria
    of the batch are solved simultaneously by a vectorized Newton iteration.

    Args:
        fs: Fluid system
        element_composition: Elemental composition with shape (..., n_elements)
        h: Absolute enthalpy in J, broadcastable to the batch shape
        p: Pressure in Pascal, broadcastable to the batch shape
        t_guess: Start value for the temperature in Kelvin, broadcastable to the batch shape

    Returns:
        Tuple of the molar amounts of species with shape (..., n_species) and
        the temperatures with shape (...), where ... is the broadcast shape of
        element_composition[..., 0], h and p. Values are nan if the temperature
        is outside of the temperature range of the fluid system.
    """
    np = fs.np
    element_composition, h, p = _broadcast_inputs(fs, element_composition, h, p)
    t_min, t_max = _t_range(fs)

    a = fs.array_stoichiometric_coefficients  # (n_reactions, n_species)
    species_elements = fs.array_species_elements  # (n_species, n_elements)
    a_sum = np.sum(a, axis=1)

    el_max = np.max(element_composition, axis=-1, keepdims=True)
    element_norm = element_composition / el_max
    element_norm_log = np.log(element_norm + epsy)
    log_p = np.log(p / p0)

    # Calculating the maximum possible amount for each species based on the elements
    species_max = np.min((element_norm[..., None, :] + epsy) / (species_elements + epsy), axis=-1)
    species_max_log = np.log(species_max + epsy)

    # The enthalpy of a composition is only meaningful if it satisfies the elemental balance,
    # therefore the start values are the isothermal equilibrium at the guessed temperature
    t_start = np.clip(np.broadcast_to(fs._asarray(t_guess), h.shape), t_min, t_max)
    logn_start = np.log(equilibrium_eq(fs, element_norm, t_start, p) + epsy)

    # Energy balance is scaled by a constant RT to keep its derivative for T positive
    rt_scale = R * t_start
    h_norm = h / el_max[..., 0] / rt_scale

    def residuals(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        logn = x[..., :-1]
        t = np.exp(x[..., -1])
        n: FloatArray = np.exp(logn)  # n is the molar amount normalized by el_max
        n_sum = np.sum(n, axis=-1, keepdims=True)
        species_h = fs.get_species_h(t)
        h_rt = species_h / (R * t[..., None])

        # Residuals from equilibrium equations with pressure corrected log equilibrium constants:
        bp = -fs.get_species_g_rt(t) @ a.T - a_sum * log_p[..., None]
        resid_eq = (logn - np.log(n_sum)) @ a.T - bp

        # Jacobian for equilibrium equations: a @ (I - 1 * n / n_sum) and
        # d(g/RT)/d(ln T) = -h/RT for the temperature
        j_eq = np.concatenate([a - a_sum[:, None] * (n / n_sum)[..., None, :],
                               -(h_rt @ a.T)[..., None]], axis=-1)

        # Residuals from elemental balance:
        el_sum_norm = n @ species_elements + epsy
        resid_ab = np.log(el_sum_norm) - element_norm_log

        # Jacobian for elemental balance (independent of temperature):
        j_ab = species_elements.T * n[..., None, :] / el_sum_norm[..., :, None]
        j_ab = np.concatenate([j_ab, np.zeros_like(j_ab[..., :1])], axis=-1)

        # Residual from energy balance:
        h_species = species_h / rt_scale[..., None]
        resid_h = np.sum(n * h_species, axis=-1) - h_norm

        # Jacobian for energy balance: n * h for the species and n * cp * T for ln T
        cp_t = np.sum(n * fs.get_species_cp(t), axis=-1) * t / rt_scale
        j_h = np.concatenate([n * h_species, cp_t[..., None]], axis=-1)[..., None, :]

        return (np.concatenate([resid_eq, resid_ab, resid_h[..., None]], axis=-1),
                np.concatenate([j_eq, j_ab, j_h], axis=-2))

    # The residuals can not get smaller than the precision of the data type
    tol = max(1e-10, 1e3 * float(np.finfo(species_max_log.dtype).eps))

    def newton_step(x: FloatArray) -> tuple[FloatArray, Any]:
        rF, J = residuals(x)

        delta = np.linalg.solve(J, -rF[..., None])[..., 0]

        # Iterate until all equilibria of the batch are converged
        converged = np.all(np.linalg.norm(rF, axis=-1) < tol)

        # The Newton step is damped to limit the temperature step to 20 %
        delta = delta * np.minimum(1, 0.2 / (np.abs(delta[..., -1:]) + epsy))

        # The temperature is limited to the range of the fluid system
        logn = np.minimum(x[..., :-1] + delta[..., :-1], species_max_log + 1)
        log_t = np.clip(x[..., -1] + delta[..., -1], np.log(t_min), np.log(t_max))
        return np.concatenate([logn, log_t[..., None]], axis=-1), converged

    x: FloatArray = iterate(newton_step, np.concatenate([logn_start, np.log(t_start)[..., None]], axis=-1), 60, np)

    # The energy balance can not be satisfied if the temperature is out of range
    valid = np.abs(residuals(x)[0][..., -1]) < max(1e-6, 10 * tol)

    n_eq: FloatArray = np.where(valid[..., None], np.exp(x[..., :-1]) * el_max, np.nan)
    t_eq: FloatArray = np.where(valid, np.exp(x[..., -1]), np.nan)
    return n_eq, t_eq


def _temperature_from_enthalpy(fs: fluid_system, composition: FloatArray, h: FloatArray,
                               t_guess: float | FloatArray) -> FloatArray:
    """Calculate the temperature of a fluid with fixed composition (..., n_species)
    at a given absolute enthalpy (...)"""
    np = fs.np
    t_min, t_max = _t_range(fs)
    t_start = np.clip(np.broadcast_to(fs._asarray(t_guess), h.shape), t_min, t_max)

    # Energy balance is scaled by a constant RT to keep its derivative for T positive
    rt_scale = R * t_start * np.sum(composition, axis=-1)

    def resid(log_t: FloatArray) -> FloatArray:
        return (np.sum(composition * fs.get_species_h(np.exp(log_t)), axis=-1) - h) / rt_scale  # type: ignore[no-any-return]

    # The residuals can not get smaller than the precision of the data type
    tol = max(1e-10, 1e3 * float(np.finfo(t_start.dtype).eps))

    def newton_step(log_t: FloatArray) -> tuple[FloatArray, Any]:
        t = np.exp(log_t)
        r = resid(log_t)
        dr = np.sum(composition * fs.get_species_cp(t), axis=-1) * t / rt_scale
        converged = np.all(np.abs(r) < tol)
        log_t = np.clip(log_t - np.clip(r / dr, -0.2, 0.2), np.log(t_min), np.log(t_max))
        return log_t, converged

    log_t: FloatArray = iterate(newton_step, np.log(t_start), 60, np)
    return np.where(np.abs(resid(log_t)) < max(1e-6, 10 * tol), np.exp(log_t), np.nan)  # type: ignore[no-any-return]


def equilibrium_hp(f: fluid | elements, h: float | FloatArray, p: float | FloatArray = 1e5,
                   t_guess: float | FloatArray = 1000.0) -> tuple[fluid, FloatArray]:
    """Calculate the isobaric equilibrium composition and temperature of a fluid
    at a given absolute enthalpy and pressure (e.g. for adiabatic reactions)

    Enthalpies are referenced like in fluid.get_H (including enthalpy of formation).
    The enthalpy of a fluid fl at temperature t is therefore fl.get_H(t). Mixing
    streams, adding heat or calculating the adiabatic flame temperature of a fluid
    are expressed by the enthalpy argument:

        >>> fl_out, t_out = gp.equilibrium_hp(fl1 + fl2, fl1.get_H(t1) + fl2.get_H(t2) + q)

    The shapes of f, h and p are broadcast against each other, so multiple
    equilibria are solved in parallel. Independent of the solver selected by
    set_solver this function uses the "system of equations" solver.

    Args:
        f: Fluid or elements object
        h: Absolute enthalpy in J
        p: Pressure in Pascal
        t_guess: Start value of the temperature iteration in Kelvin

    Returns:
        Tuple of a new fluid object with the equilibrium composition and the
        equilibrium temperatures in Kelvin. Composition and temperature are nan
        if the temperature is outside of the temperature range of the fluid
        system (t_min and t_max of fluid_system).
    """
    assert isinstance(f, (fluid, elements)), 'Argument f must be a fluid or elements'
    fs = f.fs
    if fs.array_stoichiometric_coefficients.shape[0]:
        composition, t = equilibrium_eq_hp(fs, f.array_elemental_composition, h, p, t_guess)
        return fluid(composition, fs), t

    # No reactions possible, the composition is fixed
    composition, h, _ = _broadcast_inputs(fs, equilibrium(f, t_guess).array_composition, h, p)
    return fluid(composition, fs), _temperature_from_enthalpy(fs, composition, h, t_guess)


_equilibrium_solver = equilibrium_eq
