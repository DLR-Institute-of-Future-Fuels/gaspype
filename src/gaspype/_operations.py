from ._main import T, elements, fluid, species_system, species_properties, _species_db, _condensed_species_db
from .typing import FloatArray
from .constants import p0
from ._solver import equilibrium


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
    np = a0.fs.np
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
    np = a0.fs.np
    return a0.__class__(np.concatenate(
        [a.array_elemental_composition if isinstance(a, elements) else a.array_composition for a in arrays],
        axis=axis), a0.fs)


def activity(f: fluid | elements, t: float | FloatArray, p: float | FloatArray,
             species: str | species_system) -> FloatArray:
    """Calculate the activity of a substance in the equilibrium state of a fluid
    at a given temperature and pressure. The substance does not need to be part
    of the fluid system, it can be a condensed phase species (solid or liquid,
    see gaspype.species(condensed=True)) or a gas phase species.

    For a condensed phase species the activity is relative to the pure
    substance: At a value of 1 the fluid is in equilibrium with the substance.
    At a value > 1 formation of the substance is thermodynamic favored. At a
    value < 1 a depletion of the substance is favored. For a gas phase species
    the activity is its equilibrium partial pressure divided by the standard
    pressure p0.

    The activity is derived from the chemical potentials of the elements in
    the equilibrated fluid, therefore it does not depend on a specific
    formation reaction. The shapes of f, t and p are broadcast against each
    other.

    For repeated calls a species_system with the substance as single species
    can be passed instead of its name. Its properties are then interpolated
    from the precalculated data of the species_system, like the properties of
    the fluid, instead of being calculated from the polynomials for each call.

    Args:
        f: Fluid or elements object
        t: Temperature in Kelvin
        p: Pressure in Pascal
        species: Name of the substance, e.g. 'C(gr)', 'Ni(cr)' or 'O2', or a
            species_system (or fluid_system) with a single species and the
            same backend as the fluid

    Returns:
        The activity of the substance. It is 0 if the fluid does not contain
        all elements of the substance and nan if the activity is not defined:
        For temperatures outside of the temperature range of the substance or
        if the chemical potential of an element can not be determined (e.g.
        for carbon in a fluid system with CO as only carbon containing species).
    """
    fs = f.fs
    np = fs.np
    t_arr = fs._asarray(t)
    p_arr = fs._asarray(p)

    if isinstance(species, species_system):
        assert len(species.species) == 1, 'The species_system must have a single species'
        assert species.np is np, 'The species_system must have the same backend as the fluid'
        composition = species._species_compositions[0]
        g_rt_s = species.get_species_g_rt(t_arr)[..., 0]
    else:
        species_data = _condensed_species_db.read(species) or _species_db.read(species)
        if not species_data:
            raise Exception(f'Species {species} not found')
        composition = species_data.composition

        if species in fs.species:
            g_rt_s = fs.get_species_g_rt(t_arr)[..., fs.species.index(species)]
        else:
            g_rt_s = species_properties(species_data, t_arr, np)[3]

    x = equilibrium(f, t_arr, p_arr).array_fractions
    species_elements = fs.array_species_elements  # (n_species, n_elements)

    # Species and elements below this fraction are treated as absent
    x_min = 1e-20
    present = x > x_min
    x = np.where(present, x, x_min)

    # Chemical potentials of the species divided by RT
    mu_rt = fs.get_species_g_rt(t_arr) + np.log(x) + np.log(p_arr / p0)[..., None]

    # In the equilibrium the chemical potential of each species is the sum of
    # the potentials of its elements: species_elements @ el_potential = mu_rt
    # This overdetermined system is solved by least squares, weighted by the
    # fractions since potentials of trace species are less accurate.
    w = np.where(present, np.sqrt(x), 0.0)
    m = species_elements * w[..., None]  # (..., n_species, n_elements)
    rtol = max(1e-10, 1e3 * float(np.finfo(x.dtype).eps))
    m_pinv = np.linalg.pinv(m, rtol)  # (..., n_elements, n_species)
    el_potential = (m_pinv @ (mu_rt * w)[..., None])[..., 0]

    formula = fs._asarray([float(composition.get(el, 0)) for el in fs.elements])
    el_absent = (x * present) @ species_elements <= x_min
    formula_present = np.where(el_absent, 0.0, formula)

    # The activity is only defined if the formula of the substance is a
    # linear combination of the formulas of the present species
    projection = (formula_present[..., None, :] @ (m_pinv @ m))[..., 0, :]
    defined = np.all(np.abs(projection - formula_present) < 1e-2, axis=-1)

    a = np.where(defined, np.exp(np.sum(formula_present * el_potential, axis=-1) - g_rt_s), np.nan)

    if not set(composition).issubset(fs.elements):
        return a * 0.0  # type: ignore[no-any-return]

    # nan is kept for temperatures outside of the range of the substance
    return np.where(np.any(el_absent & (formula > 0), axis=-1) & (g_rt_s == g_rt_s), 0.0, a)  # type: ignore[no-any-return]


def carbon_activity(f: fluid | elements, t: float | FloatArray, p: float | FloatArray) -> FloatArray:
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
    return activity(f, t, p, 'C(gr)')


def oxygen_partial_pressure(f: fluid | elements, t: float | FloatArray, p: float | FloatArray) -> FloatArray:
    """Calculate the oxygen partial pressure in the equilibrium state of a
    fluid at a given temperature and pressure. The fluid system does not
    need to include O2.

    Args:
        f: Fluid or elements object
        t: Temperature in Kelvin
        p: Pressure in Pascal

    Returns:
        The oxygen partial pressure in Pascal
    """
    return activity(f, t, p, 'O2') * p0
