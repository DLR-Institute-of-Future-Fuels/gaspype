from typing import Literal, Any
from math import exp
from scipy.optimize import minimize, root
import numpy as np
from ._main import T, NDFloat, elements, fluid, fluid_system, FloatArray
from .constants import p0, epsy, R


def set_solver(solver: Literal['gibs minimization', 'system of equations']) -> None:
    """
    Select a solver for chemical equilibrium.

    Solvers:
        - **system of equations** (default): Finds the root for a system of
          equations covering a minimal set of equilibrium equations and elemental balance.
          The minimal set of equilibrium equations is derived by SVD using the null_space
          implementation of scipy.

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


def stack(arrays: list[T], axis: int = 0) -> T:
    """Stack a list of fluid or elements objects along a new axis

    Args:
        arrays: List of arrays
        axis: Axis to stack the fluid objects along

    Returns:
        A new array object stacked along the new axis
    """
    a0 = arrays[0]
    assert all(a.fs == a0.fs for a in arrays), 'All objects must have the same fluid system'
    assert axis <= len(a0.shape), f'Axis must be smaller or equal to len(shape) ({len(a0.shape)})'
    return a0.__class__(np.stack(
        [a.array_elemental_composition if isinstance(a, elements) else a.array_composition for a in arrays],
        axis=axis), a0.fs)


def concat(arrays: list[T], axis: int = 0) -> T:
    """Concatenate a list of fluid or elements objects along an existing axis

    Args:
        arrays: List of arrays
        axis: Axis to concatenate the fluid objects along

    Returns:
        A new array object stacked along the specified axis
    """
    a0 = arrays[0]
    assert all(f.fs == a0.fs for f in arrays), 'All fluid objects must have the same fluid system'
    assert axis < len(a0.shape), f'Axis must be smaller than shape len({a0.shape})'
    return a0.__class__(np.concatenate(
        [a.array_elemental_composition if isinstance(a, elements) else a.array_composition for a in arrays],
        axis=axis), a0.fs)


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

    a = fs.array_stoichiometric_coefficients
    a_sum = np.sum(a)
    el_matrix = fs.array_species_elements.T

    # Log equilibrium constants for each reaction equation
    b = -np.sum(fs.get_species_g_rt(t) * a, axis=1)

    # Pressure corrected log equilibrium constants
    bp = b - np.sum(a * np.log(p / p0), axis=1)

    logn_start = np.ones(el_matrix.shape[1]) * 0.1

    def residuals(logn: FloatArray):  # type: ignore
        n = np.exp(logn)
        n_sum = np.sum(n)

        # Residuals from equilibrium equations:
        eq_resid = np.dot(a, logn - np.log(n_sum)) - bp

        # Derivative:
        j_eq = a - a_sum * n / n_sum

        # Residuals from elemental balance:
        el_error = np.dot(el_matrix, n) - element_norm
        ab_resid = np.log1p(el_error)

        # Derivative:
        j_ab = el_matrix * n / np.expand_dims(el_error + 1, axis=1)

        return (np.hstack([eq_resid, ab_resid]), np.concatenate([j_eq, j_ab], axis=0))

    ret = root(residuals, logn_start, jac=True, tol=1e-30)
    n = np.exp(np.array(ret['x'], dtype=NDFloat))

    return n * el_max


def equilibrium(f: fluid | elements, t: float | FloatArray, p: float = 1e5) -> fluid:
    """Calculate the equilibrium composition of a fluid at a given temperature and pressure"

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

            print('-->', f.array_elemental_composition.shape, f.fs.array_species_elements.transpose().shape)
            composition = np.apply_along_axis(linalg_lstsq, -1, f.array_elemental_composition, f.fs.array_species_elements.transpose())
            return fluid(composition, f.fs)

    assert np.min(f.array_elemental_composition) >= 0, 'Input element fractions must be 0 or positive'
    if isinstance(t, np.ndarray):
        assert f.shape == tuple(), 'Multidimensional temperature can currently only used for 0D fluids'
        t_composition = np.zeros(t.shape + (f.fs.array_species_elements.shape[0],))
        for t_index in np.ndindex(t.shape):
            t_composition[t_index] = _equilibrium_solver(f.fs, f.array_elemental_composition, t[t_index], p)
        return fluid(t_composition, f.fs)
    else:
        composition = np.ones(f.shape + (len(f.fs.species),), dtype=float)
        for index in np.ndindex(f.shape):
            #print(composition.shape, index, _equilibrium(f.fs, f._element_composition[index], t, p))
            composition[index] = _equilibrium_solver(f.fs, f.array_elemental_composition[index], t, p)
        return fluid(composition, f.fs)


def carbon_activity(f: fluid | elements, t: float, p: float) -> float:
    """Calculate the activity of carbon in a fluid at a given temperature and pressure.
    At a value of 1 the fluid is in equilibrium with solid graphite. At a value > 1
    additional carbon formation is thermodynamic favored. At a value < 1 a
    depletion of solid carbon is favored.

    Args:
        f: Fluid or elements object
        t: Temperature in Kelvin
        p: Pressure in Pascal

    Returns:
        The activity of carbon in the fluid
    """
    # Values for solid state carbon (graphite) from NIST-JANAF Tables
    # https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Carbon.pdf
    # https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-1Vol1-Intro.pdf
    # Polynomial is valid for T from 100 to 2500 K
    pgef = np.array([-6.76113852E-02, 2.02187857E+00, -2.38664605E+01,
                    1.43575462E+02, -4.51375503E+02, 6.06219665E+02])

    # Gibbs free energy divided by RT for carbon
    g_rtc = -np.sum(pgef * np.log(np.expand_dims(t, -1))**np.array([5, 4, 3, 2, 1, 0])) / R

    g_rt = f.fs.get_species_g_rt(t)

    x = equilibrium(f, t, p).array_fractions

    i_co = f.fs.species.index('CO')
    i_co2 = f.fs.species.index('CO2')
    i_h2 = f.fs.species.index('H2')
    i_h2o = f.fs.species.index('H2O')
    i_ch4 = f.fs.species.index('CH4')

    if min(x[i_co], x[i_co2]) > min([x[i_ch4], x[i_h2o], x[i_h2]]) and min(x[i_co], x[i_co2]) > 0:
        # 2 CO -> CO2 + C(s) (Boudouard reaction)
        lnalpha = (2 * g_rt[i_co] - (g_rt[i_co2] + g_rtc)) + np.log(
            x[i_co]**2 / x[i_co2] * (p / p0))
    elif min([x[i_ch4], x[i_h2o], x[i_co]]) > 1E-4:
        # CH4 + 2 CO -> 2 H2O + 3 C(s)
        lnalpha = ((g_rt[i_ch4] + 2 * g_rt[i_co] - 3 * g_rtc - 2 * g_rt[i_h2o]) + np.log(
            x[i_ch4] * x[i_co]**2 / x[i_h2o]**2 * (p / p0))) / 3
    elif min(x[i_h2], x[i_ch4]) > 0:
        # if x[i_h2] or x[i_ch4] is small compared to the precision of the
        # component concentrations the result can be inaccurate
        # CH4 -> 2 H2 + C(s)
        # CH4 + CO2 -> 2 H2 + 2 CO
        # 2 H2O - O2 -> 2 H2
        lnalpha = (g_rt[i_ch4] - (2 * g_rt[i_h2] + g_rtc)) + np.log(
            x[i_ch4] / x[i_h2]**2 / (p / p0))
    elif x[i_h2] == 0:
        # equilibrium on carbon side
        lnalpha = 10
    else:
        # equilibrium on non-carbon side
        lnalpha = -10

    return exp(lnalpha)


def oxygen_partial_pressure(f: fluid | elements, t: float, p: float) -> FloatArray | float:
    _oxygen_data = fluid({'O2': 1})

    def get_oxygen(x: FloatArray) -> float:
        g_rt = f.fs.get_species_g_rt(t)
        g_rt_o2 = _oxygen_data.fs.get_species_g_rt(t)[0]

        i_co = f.fs.species.index('CO') if 'C' in f.fs.elements else None
        i_co2 = f.fs.species.index('CO2') if 'C' in f.fs.elements else None
        i_o2 = f.fs.species.index('O2') if 'O2' in f.fs.species else None
        i_h2o = f.fs.species.index('H2O') if 'H' in f.fs.elements else None
        i_h2 = f.fs.species.index('H2') if 'H' in f.fs.elements else None
        i_ch4 = f.fs.species.index('CH4') if 'CH4' in f.fs.species else None

        ox_ref_val = max([float(v) for v in (x[i_h2] * x[i_h2o], x[i_co2] * x[i_co], x[i_ch4]) if v.shape == tuple()] + [0])

        # print([i_o2, x[i_o2], ox_ref_val])

        if i_o2 is not None and x[i_o2] > ox_ref_val:
            # print('o O2')
            return float(x[i_o2] * p)

        elif i_ch4 is not None and x[i_ch4] > x[i_co2] * 100 and x[i_ch4] > x[i_h2o] * 100:
            # print('o ch4')
            # 2CH4 + O2 <--> 4H2 + 2CO
            lnpo2 = 4 * g_rt[i_h2] + 2 * g_rt[i_co] - 2 * g_rt[i_ch4] - g_rt_o2 + np.log(x[i_h2]**4 * x[i_co]**2 / x[i_ch4]**2) - 2 * np.log(p / p0)

        elif (i_co is None and i_h2 is not None) or (i_h2 is not None and i_co is not None and (x[i_h2] * x[i_h2o] > x[i_co2] * x[i_co])):
            # print('o h2o')
            # 2H2 + O2 <--> 2H2O
            lnpo2 = 2 * (g_rt[i_h2o] - g_rt[i_h2] + np.log(x[i_h2o] / x[i_h2])) - g_rt_o2 - np.log(p / p0)

        else:
            assert i_co is not None
            # print('o co2')
            # 2CO + O2 <--> 2CO2
            lnpo2 = 2 * (g_rt[i_co2] - g_rt[i_co] + np.log(x[i_co2] / x[i_co])) - g_rt_o2 - np.log(p / p0)

        return exp(lnpo2) * p

    x = equilibrium(f, t, p).array_fractions

    if len(x.shape):
        return np.apply_along_axis(get_oxygen, -1, x)
    else:
        return get_oxygen(x)


_equilibrium_solver = equilibrium_eq
