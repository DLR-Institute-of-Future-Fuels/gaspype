import gaspype as gp
import numpy as np
import pytest
from gaspype.constants import p0


fs = gp.fluid_system('CO, CO2, H2, H2O, CH4, N2, O2')


def test_gas_species_activity():
    # The activity of a gas phase species is its partial pressure divided by p0
    fl = gp.fluid({'CH4': 1, 'H2O': 1.5, 'N2': 0.1}, fs)
    t = 800 + 273.15
    p = 5e5

    eq = gp.equilibrium(fl, t, p)

    for s in fs.species:
        assert gp.activity(fl, t, p, s) * p0 == pytest.approx(eq.get_x(s) * p, rel=1e-6)


def test_species_not_in_fluid_system():
    # The result must not depend on the substance being part of the fluid system
    fs_without_o2 = gp.fluid_system('CO, CO2, H2, H2O, CH4, N2')
    composition = {'CH4': 1, 'CO2': 0.2, 'N2': 1}
    t = 1100 + 273.15
    p = 20e5

    o2_ref = gp.equilibrium(gp.fluid(composition, fs), t, p).get_x('O2') * p

    assert gp.oxygen_partial_pressure(gp.fluid(composition, fs_without_o2), t, p) == pytest.approx(o2_ref, rel=1e-4)
    assert gp.oxygen_partial_pressure(gp.fluid(composition, fs), t, p) == pytest.approx(o2_ref, rel=1e-6)


def test_carbon_activity_reactions():
    # Compare with the carbon activities derived from specific reactions
    fl = gp.fluid({'CH4': 1, 'H2O': 1.2}, fs)
    t = 700 + 273.15
    p = 3e5

    x = gp.equilibrium(fl, t, p)
    g_rt = dict(zip(fs.species, fs.get_species_g_rt(t)))
    g_rt_c = np.log(gp.activity(gp.fluid({'CH4': 1}, fs), t, p0, 'CH4')) + g_rt['CH4'] - 2 * (
        np.log(gp.activity(gp.fluid({'CH4': 1}, fs), t, p0, 'H2')) + g_rt['H2'])
    g_rt_c -= np.log(gp.carbon_activity(gp.fluid({'CH4': 1}, fs), t, p0))

    # 2 CO -> CO2 + C(s) (Boudouard reaction)
    boudouard = np.exp(2 * g_rt['CO'] - g_rt['CO2'] - g_rt_c) * x.get_x('CO')**2 / x.get_x('CO2') * p / p0
    # CH4 -> 2 H2 + C(s)
    cracking = np.exp(g_rt['CH4'] - 2 * g_rt['H2'] - g_rt_c) * x.get_x('CH4') / x.get_x('H2')**2 / (p / p0)

    assert gp.carbon_activity(fl, t, p) == pytest.approx(boudouard, rel=1e-6)
    assert gp.carbon_activity(fl, t, p) == pytest.approx(cracking, rel=1e-6)


def test_graphite_boudouard_reference():
    # Boudouard equilibrium 2 CO <-> CO2 + C(gr) at 1 bar: the gas is in
    # equilibrium with graphite (activity = 1) at about 700 degC for
    # x_CO = 0.6, x_CO2 = 0.4 (JANAF: K_p(973 K) = 1.07)
    fs_co = gp.fluid_system('CO, CO2')
    fl = gp.fluid({'CO': 0.6, 'CO2': 0.4}, fs_co)

    assert gp.carbon_activity(fl, 973.15, p0) == pytest.approx(1, rel=0.2)
    assert gp.carbon_activity(fl, 873.15, p0) > 5
    assert gp.carbon_activity(fl, 1073.15, p0) < 0.2


def test_water_saturation():
    # The activity of liquid water is p_H2O / p_sat
    fl = gp.fluid({'H2O': 1, 'N2': 1}, gp.fluid_system('H2O, N2'))

    assert gp.activity(fl, 25 + 273.15, 2 * 3170, 'H2O(L)') == pytest.approx(1, rel=0.01)
    assert gp.activity(fl, 100 + 273.15, 2 * 101325, 'H2O(L)') == pytest.approx(1, rel=0.03)
    assert gp.activity(fl, 100 + 273.15, 101325, 'H2O(L)') == pytest.approx(0.5, rel=0.03)


def test_broadcasting():
    ratio = np.linspace(0.5, 3, 4)
    fl = gp.fluid({'CH4': 1}, fs) + ratio[:, None] * gp.fluid({'H2O': 1}, fs)
    t = np.array([600, 700, 800]) + 273.15
    p = 1e5

    result = gp.carbon_activity(fl, t, p)
    assert result.shape == (4, 3)

    for i in range(4):
        for j in range(3):
            assert result[i, j] == pytest.approx(gp.carbon_activity(fl[i, 0], t[j], p), rel=1e-9)

    assert gp.carbon_activity(gp.elements(fl), t, p) == pytest.approx(result, rel=1e-9)


def test_absent_elements():
    fl = gp.fluid({'H2O': 1, 'H2': 1}, fs)

    # No carbon in the fluid
    assert gp.carbon_activity(fl, 1000, 1e5) == 0
    # Element is not part of the fluid system
    assert gp.activity(fl, 1000, 1e5, 'Ni(cr)') == 0
    assert np.all(gp.activity(fl, np.array([900, 1000]), 1e5, 'Ni(cr)') == 0)


def test_undefined_activity():
    fl = gp.fluid({'H2O': 1, 'H2': 1, 'CO': 0.1}, fs)

    result = gp.activity(fl, np.array([1000., 1800.]), 1e5, 'C(gr)')
    assert np.all(np.isfinite(result))

    # The data for H2O(L) is only valid from 273.15 K to 600 K
    result = gp.activity(fl, np.array([400., 700.]), 1e5, 'H2O(L)')
    assert np.isfinite(result[0]) and np.isnan(result[1])

    # Potentials of C and O can not be separated with CO as only species of both
    fl_co = gp.fluid({'CO': 1, 'N2': 1})
    assert np.isnan(gp.carbon_activity(fl_co, 1000, 1e5))
    assert gp.activity(fl_co, 1000, 1e5, 'CO') == pytest.approx(0.5)
    assert gp.activity(fl_co, 1000, 1e5, 'N2') == pytest.approx(0.5)


def test_unknown_species():
    with pytest.raises(Exception, match='not found'):
        gp.activity(gp.fluid({'H2': 1}), 1000, 1e5, 'Xx(cr)')


def test_condensed_species_list():
    condensed = gp.species(condensed=True)

    assert 'C(gr)' in condensed
    assert 'C(gr)' not in gp.species()
    assert gp.species('C*', 'C', condensed=True) == ['C(gr)']
    assert set(gp.species(element_names='Ni', condensed=True)) == {'Ni(L)', 'Ni(cr)'}


def test_species_system():
    fl = gp.fluid({'CH4': 1, 'H2O': 1.2, 'N2': 0.1}, fs)
    t = np.array([400.5, 800.25, 1300.0])
    p = 3e5

    # Condensed and gas phase species can be mixed
    ss = gp.species_system('C(gr), H2O(L), O2')
    assert ss.species == ['C(gr)', 'H2O(L)', 'O2']
    assert ss.elements == ['C', 'H', 'O']
    assert 'Graphite' in ss.get_species_references()
    assert ss.get_species_g_rt(t).shape == (3, 3)

    for name in ['C(gr)', 'O2', 'CO']:
        ref = gp.activity(fl, t, p, name)
        assert gp.activity(fl, t, p, gp.species_system(name)) == pytest.approx(ref, rel=1e-5)

    # A fluid_system with a single species is a species_system as well
    assert gp.activity(fl, t, p, gp.fluid_system('O2')) == pytest.approx(gp.activity(fl, t, p, 'O2'), rel=1e-5)

    # Outside of the temperature range of the substance (273.15 K to 600 K)
    result = gp.activity(fl, np.array([400.5, 700.0]), p, gp.species_system('H2O(L)'))
    assert np.isfinite(result[0]) and np.isnan(result[1])

    # Element of the substance is not in the fluid
    assert gp.activity(fl, 1000, p, gp.species_system('Ni(cr)')) == 0

    with pytest.raises(AssertionError):
        gp.activity(fl, t, p, ss)

    # Condensed species can not be part of a fluid
    with pytest.raises(Exception, match='not found'):
        gp.fluid_system('H2, C(gr)')
