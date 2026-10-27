import gaspype as gp
import numpy as np
import pytest

fs = gp.fluid_system('CH4, H2O, H2, CO2, CO, O2, N2', t_max=3500)


def test_energy_balance_and_isothermal_consistency():
    phi = np.linspace(0.3, 3, 20)
    fl = gp.fluid({'CH4': 1}, fs) * phi[:, None] + gp.fluid({'O2': 2, 'N2': 7.52}, fs)
    t_in = np.array([300.0, 600.0, 900.0])
    h = fl.get_H(t_in)

    fl_out, t_out = gp.equilibrium_hp(fl, h, 1e5)

    assert fl_out.shape == (20, 3)
    assert t_out.shape == (20, 3)
    assert np.all(t_out > t_in)
    assert fl_out.get_H(t_out) == pytest.approx(h, rel=1e-9)

    # The result is the isothermal equilibrium at the resulting temperature
    fl_tp = gp.equilibrium(fl, t_out, 1e5)
    assert fl_out.array_composition == pytest.approx(fl_tp.array_composition, rel=1e-6, abs=1e-12)


def test_mixing_and_heat_input():
    fuel = gp.fluid({'CH4': 1, 'H2O': 2}, fs)
    oxidant = gp.fluid({'O2': 0.5}, fs)
    q = 50e3  # J

    fl_out, t_out = gp.equilibrium_hp(fuel + oxidant, fuel.get_H(500) + oxidant.get_H(800) + q, 2e5)

    assert float(fl_out.get_H(t_out)) == pytest.approx(float(fuel.get_H(500) + oxidant.get_H(800) + q), rel=1e-9)
    assert float(t_out) == pytest.approx(float(gp.equilibrium_hp(fuel + oxidant, fl_out.get_H(t_out), 2e5)[1]))


def test_elements_input_and_start_value():
    fl = gp.fluid({'CH4': 1, 'H2O': 1.5}, fs)
    h = fl.get_H(1100)

    _, t_ref = gp.equilibrium_hp(fl, h)
    for t_guess in [300, 1000, 3000]:
        fl_out, t_out = gp.equilibrium_hp(gp.elements(fl), h, t_guess=t_guess)
        assert float(t_out) == pytest.approx(float(t_ref), rel=1e-9)
        assert float(fl_out.get_H(t_out)) == pytest.approx(float(h), rel=1e-9)


def test_temperature_out_of_range():
    fs_small = gp.fluid_system('H2, H2O, O2', t_max=2000)
    fl = gp.fluid({'H2': 2, 'O2': 1}, fs_small) * np.array([1, 0.01])[:, None] + gp.fluid({'H2O': 1}, fs_small)

    fl_out, t_out = gp.equilibrium_hp(fl, fl.get_H(300))

    # Stoichiometric hydrogen combustion exceeds t_max, the diluted mixture not
    assert np.isnan(t_out[0]) and np.all(np.isnan(fl_out.array_composition[0]))
    assert 300 < t_out[1] < 2000


def test_no_reactions():
    fs_inert = gp.fluid_system('N2, Ar')
    fl = gp.fluid({'N2': 1, 'Ar': 0.5}, fs_inert)
    t = np.array([400.0, 900.0])

    fl_out, t_out = gp.equilibrium_hp(fl, fl.get_H(t), np.array([[1e5], [3e5]]))

    assert fl_out.shape == (2, 2)
    assert t_out == pytest.approx(np.array([t, t]), rel=1e-9)
    assert fl_out.array_composition[1, 0] == pytest.approx(fl.array_composition)
