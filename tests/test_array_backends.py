import gaspype as gp
import numpy as np
import pytest
import warnings

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
jnp = jax.numpy

species = 'CH4, H2O, H2, CO2, CO, O2, N2'
fs = gp.fluid_system(species)
fs_jax = gp.fluid_system(species, backend='jax')
composition = np.array([[1, 2, 0, 0, 0, 0, 0.5], [0.5, 1, 0.2, 0, 0.1, 0, 0]])
t = np.array([900.0, 1300.0])
p = 2e5


def is_same(result, reference):
    """Check that result is a JAX array with the values of the NumPy reference"""
    assert isinstance(result, jax.Array), f'{type(result)} is not a JAX array'
    return np.asarray(result) == pytest.approx(reference, rel=1e-9, abs=1e-30)


def test_backend_selection():
    assert fs.np is np
    assert fs.device is None
    assert isinstance(gp.fluid(composition, fs).get_h(t), np.ndarray)

    for backend in ['jax', 'jax.numpy', jnp]:
        fs_backend = gp.fluid_system('H2, H2O, O2', backend=backend)
        assert fs_backend.np is jnp
        assert isinstance(fs_backend.array_molar_mass, jax.Array)
        assert isinstance(fs_backend.get_species_h(900), jax.Array)

    # Fluid systems derived from a fluid system keep its backend
    assert (fs_jax + gp.fluid_system('Ar', backend='jax')).np is jnp

    with pytest.raises(ValueError):
        gp.fluid_system('H2, H2O, O2', device='cpu')


def test_device():
    cpu = jax.devices('cpu')[0]

    for device in ['cpu', 'cpu:0', cpu]:
        fs_cpu = gp.fluid_system(species, backend='jax', device=device)
        assert fs_cpu.device == cpu
        assert fs_cpu.array_molar_mass.device == cpu

    ref = gp.fluid(composition, fs)
    fl = gp.fluid(composition, fs_cpu)
    assert fl.array_composition.device == cpu
    assert is_same(fl.get_s(t, p), ref.get_s(t, p))
    assert is_same(gp.equilibrium(fl, t, p).array_composition, gp.equilibrium(ref, t, p).array_composition)
    assert (fl + gp.fluid({'Ar': 1})).fs.device == cpu

    # Traced arrays
    assert is_same(jax.jit(fl.get_h)(jnp.asarray(t)), ref.get_h(t))
    assert is_same(jax.jit(lambda t: gp.equilibrium(fl, t, p).array_composition)(jnp.asarray(t)),
                   gp.equilibrium(ref, t, p).array_composition)


def test_fluid_properties():
    ref = gp.fluid(composition, fs)

    # The backend of the fluid system determines the type of the results,
    # independent of the type of the arguments
    for comp, temp in [(composition, t), (jnp.asarray(composition), jnp.asarray(t)), (composition.tolist(), t)]:
        fl = gp.fluid(comp, fs_jax)

        assert is_same(fl.array_composition, ref.array_composition)
        assert is_same(fl.array_fractions, ref.array_fractions)
        assert is_same(fl.array_elemental_composition, ref.array_elemental_composition)
        assert is_same(fl.total, ref.total)
        assert is_same(fl.get_h(temp), ref.get_h(t))
        assert is_same(fl.get_H(temp), ref.get_H(t))
        assert is_same(fl.get_s(temp, p), ref.get_s(t, p))
        assert is_same(fl.get_S(temp, p), ref.get_S(t, p))
        assert is_same(fl.get_cp(temp), ref.get_cp(t))
        assert is_same(fl.get_g(temp, p), ref.get_g(t, p))
        assert is_same(fl.get_G(temp, p), ref.get_G(t, p))
        assert is_same(fl.get_g_rt(temp, p), ref.get_g_rt(t, p))
        assert is_same(fl.get_v(temp, p), ref.get_v(t, p))
        assert is_same(fl.get_vm(temp, p), ref.get_vm(t, p))
        assert is_same(fl.get_density(temp, p), ref.get_density(t, p))
        assert is_same(fl.get_mass(), ref.get_mass())
        assert is_same(fl.get_molar_mass(), ref.get_molar_mass())
        assert is_same(fl.get_x(['H2O', 'CH4']), ref.get_x(['H2O', 'CH4']))
        assert is_same(fl.get_n('H2O'), ref.get_n('H2O'))
        assert is_same(fl['CH4'], ref['CH4'])
        assert is_same(fl[1].array_composition, ref[1].array_composition)

    # Scalar arguments
    assert is_same(fl.get_h(900), ref.get_h(900))
    assert is_same(fl.get_s(900, p), ref.get_s(900, p))
    assert is_same(fl.get_vm(900, p), ref.get_vm(900, p))

    # Fluids defined by a dict
    fl = gp.fluid({'H2O': 1, 'CH4': 2}, fs_jax, shape=(2, 3))
    ref = gp.fluid({'H2O': 1, 'CH4': 2}, fs, shape=(2, 3))
    assert is_same(fl.get_cp(900), ref.get_cp(900))
    assert repr(fl[0, 0]) == repr(ref[0, 0])
    assert [list(d) for d in fl[0]] == [list(d) for d in ref[0]]


def test_fluid_operations():
    ref = gp.fluid(composition, fs)
    fl = gp.fluid(composition, fs_jax)

    assert is_same((fl * 2).array_composition, (ref * 2).array_composition)
    assert is_same((2 * fl).array_composition, (ref * 2).array_composition)
    assert is_same((fl * t).array_composition, (ref * t).array_composition)
    assert is_same((fl * jnp.asarray(t)).array_composition, (ref * t).array_composition)
    assert is_same((fl / t).array_composition, (ref / t).array_composition)
    assert is_same((-fl).array_composition, (-ref).array_composition)
    assert is_same((fl + fl).array_composition, (ref + ref).array_composition)
    assert is_same((fl - fl[0]).array_composition, (ref - ref[0]).array_composition)
    assert is_same(gp.stack([fl, fl]).array_composition, gp.stack([ref, ref]).array_composition)
    assert is_same(gp.concat([fl, fl]).array_composition, gp.concat([ref, ref]).array_composition)


def test_operations_with_different_fluid_systems():
    ref = gp.fluid(composition, fs)
    fl = gp.fluid(composition, fs_jax)

    # Subset of the species of the fluid system
    for backend in ['numpy', 'jax']:
        sub = gp.fluid({'N2': 0.3, 'H2': 1.0}, gp.fluid_system('N2, H2', backend=backend))
        assert is_same((fl - sub).array_composition, (ref - sub).array_composition)
        assert is_same((sub + fl).array_composition, (ref + sub).array_composition)

    # Species not included in the fluid system
    other_ref = gp.fluid({'H2': 1.0, 'Ar': 0.3})
    other = gp.fluid({'H2': 1.0, 'Ar': 0.3}, gp.fluid_system('H2, Ar', backend='jax'))
    assert (fl + other).fs.species == (ref + other_ref).fs.species
    assert (fl + other).fs.np is jnp
    assert is_same((fl + other).array_composition, (ref + other_ref).array_composition)
    assert is_same((gp.elements(fl) + other).array_elemental_composition,
                   (gp.elements(ref) + other_ref).array_elemental_composition)


def test_elements():
    ref_fl = gp.fluid(composition, fs)
    ref = gp.elements(ref_fl)
    fl = gp.fluid(composition, fs_jax)
    el = gp.elements(fl)

    assert is_same(el.array_elemental_composition, ref.array_elemental_composition)
    assert is_same(gp.elements(ref.array_elemental_composition, fs_jax).get_mass(), ref.get_mass())
    assert is_same(gp.elements({'H': 2, 'O': 1}, fs_jax, shape=(3,)).get_mass(),
                   gp.elements({'H': 2, 'O': 1}, fs, shape=(3,)).get_mass())
    assert is_same(el.get_n(['O', 'H']), ref.get_n(['O', 'H']))
    assert is_same(el['C'], ref['C'])
    assert is_same((el * t).array_elemental_composition, (ref * t).array_elemental_composition)
    assert is_same((el / t).array_elemental_composition, (ref / t).array_elemental_composition)
    assert is_same((el + fl).array_elemental_composition, (ref + ref_fl).array_elemental_composition)
    assert np.asarray(el) == pytest.approx(np.asarray(ref))

    # Elements based on a fluid of another fluid system
    fs_other = gp.fluid_system('H2O, CH4, CO2, N2, Ar', backend='jax')
    assert is_same(gp.elements(ref_fl, fs_other).array_elemental_composition,
                   gp.elements(ref_fl, gp.fluid_system(fs_other.species)).array_elemental_composition)


def test_equilibrium():
    ref = gp.fluid(composition, fs)
    fl = gp.fluid(composition, fs_jax)

    assert is_same(gp.equilibrium(fl, t, p).array_composition,
                   gp.equilibrium(ref, t, p).array_composition)
    assert is_same(gp.equilibrium(fl, jnp.asarray(t), p).array_composition,
                   gp.equilibrium(ref, t, p).array_composition)
    assert is_same(gp.equilibrium(fl, 1000).array_composition,
                   gp.equilibrium(ref, 1000).array_composition)
    assert is_same(gp.equilibrium(gp.elements(fl), t, p).array_composition,
                   gp.equilibrium(gp.elements(ref), t, p).array_composition)


def test_equilibrium_without_reactions():
    el_composition = np.array([[2.0, 1.0, 0.5], [1.0, 0.0, 3.0]])
    ref = gp.elements(el_composition, gp.fluid_system('H2, N2, Ar'))
    el = gp.elements(el_composition, gp.fluid_system('H2, N2, Ar', backend='jax'))

    assert is_same(gp.equilibrium(el, 1000).array_composition,
                   gp.equilibrium(ref, 1000).array_composition)


def test_single_precision():
    ref = gp.fluid(composition, fs)

    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.simplefilter('error')
        fs_f32 = gp.fluid_system(species, backend='jax')
        fl = gp.fluid(composition, fs_f32)
        results = [fl.get_h(t), fl.get_s(t, p), fl.get_g(t, p), fl.get_density(t, p),
                   (fl * t + fl).array_composition, gp.equilibrium(fl, t, p).array_composition]

    references = [ref.get_h(t), ref.get_s(t, p), ref.get_g(t, p), ref.get_density(t, p),
                  (ref * t + ref).array_composition, gp.equilibrium(ref, t, p).array_composition]

    for result, reference in zip(results, references):
        assert result.dtype == np.float32
        assert np.asarray(result) == pytest.approx(reference, rel=1e-4, abs=1e-5)


def test_jax_transformations():
    ref = gp.fluid(composition[0], fs)
    fl = gp.fluid(composition[0], fs_jax)
    t_jax = jnp.asarray(t)

    # cp is the derivative of h with respect to the temperature
    cp = jax.jit(jax.vmap(jax.grad(fl.get_h)))(t_jax)
    assert np.asarray(cp) == pytest.approx(np.asarray(ref.get_cp(t)), rel=1e-3)

    assert is_same(jax.jit(fl.get_h)(t_jax), ref.get_h(t))
    assert is_same(jax.jit(fl.get_s)(t_jax, p), ref.get_s(t, p))

    # Fluids created from traced compositions
    def get_cp(composition, t):
        return gp.fluid(composition, fs_jax).get_cp(t)

    assert is_same(jax.jit(get_cp)(jnp.asarray(composition), t_jax), gp.fluid(composition, fs).get_cp(t))

    # Fluid systems created while tracing
    def get_h_mix(t):
        return (fl + gp.fluid({'Ar': 1}, gp.fluid_system('Ar', backend='jax'))).get_h(t)

    assert is_same(jax.jit(get_h_mix)(t_jax), (ref + gp.fluid({'Ar': 1})).get_h(t))


def test_jax_equilibrium_transformations():
    reference = gp.equilibrium(gp.fluid(composition[0], fs), t, p).array_composition
    fl = gp.fluid(composition[0], fs_jax)

    def equilibrium_composition(t):
        return gp.equilibrium(fl, t, p).array_composition

    t_jax = jnp.asarray(t)
    assert is_same(jax.jit(equilibrium_composition)(t_jax), reference)
    assert is_same(jax.vmap(equilibrium_composition)(t_jax), reference)
    assert is_same(jax.jit(jax.vmap(equilibrium_composition))(t_jax), reference)

    # Derivative of the composition with respect to the temperature. The
    # temperature is between two base points of the piecewise linear property data.
    t0, dt = 900.5, 1e-3
    finite_difference = (equilibrium_composition(t0 + dt) - equilibrium_composition(t0 - dt)) / (2 * dt)
    for jacobian in [jax.jacrev(equilibrium_composition), jax.jit(jax.jacfwd(equilibrium_composition))]:
        assert np.asarray(jacobian(jnp.asarray(t0))) == pytest.approx(finite_difference, rel=1e-5, abs=1e-9)


def test_activity():
    ref = gp.fluid(composition, fs)
    fl = gp.fluid(composition, fs_jax)

    assert is_same(gp.activity(fl, t, p, 'C(gr)'), gp.activity(ref, t, p, 'C(gr)'))
    assert is_same(gp.carbon_activity(fl, t, p), gp.carbon_activity(ref, t, p))
    assert is_same(gp.oxygen_partial_pressure(fl, t, p), gp.oxygen_partial_pressure(ref, t, p))

    t_jax = jnp.asarray(t)
    assert is_same(jax.jit(lambda t: gp.carbon_activity(fl, t, p))(t_jax), gp.carbon_activity(ref, t, p))

    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.simplefilter('error')
        fl_f32 = gp.fluid(composition, gp.fluid_system(species, backend='jax'))
        results = [gp.carbon_activity(fl_f32, t, p), gp.oxygen_partial_pressure(fl_f32, t, p)]

    for result, reference in zip(results, [gp.carbon_activity(ref, t, p), gp.oxygen_partial_pressure(ref, t, p)]):
        assert result.dtype == np.float32
        assert np.asarray(result) == pytest.approx(reference, rel=1e-3)


def test_activity_species_system():
    ref = gp.fluid(composition, fs)
    fl = gp.fluid(composition, fs_jax)
    carbon = gp.species_system('C(gr)', backend='jax')

    assert isinstance(carbon.get_species_g_rt(t), jax.Array)
    assert is_same(gp.activity(fl, t, p, carbon), gp.activity(ref, t, p, gp.species_system('C(gr)')))
    assert is_same(jax.jit(lambda t: gp.activity(fl, t, p, carbon))(jnp.asarray(t)),
                   gp.activity(ref, t, p, gp.species_system('C(gr)')))

    with pytest.raises(AssertionError):
        gp.activity(ref, t, p, carbon)
