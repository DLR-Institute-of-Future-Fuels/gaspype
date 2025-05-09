import gaspype as gp
# import gaspype.membrane as mb
import pytest


def test_oxygen_partial_pressure():
    # Compare equalibrium calculation with oxygen_partial_pressure function
    fs = gp.fluid_system('CO, CO2, H2, O2, H2O, N2')
    fl = gp.fluid({'H2O': 0.99, 'H2': 0.01}, fs)

    t = 2500

    ox1 = gp.equilibrium(fl, t, 1e5).get_x('O2') * 1e5  # Pa
    print(ox1)

    ox2 = gp.oxygen_partial_pressure(fl, t, 1e5)
    print(ox2)

    assert ox1 == pytest.approx(ox2, abs=1e-1)  # type: ignore


"""
def test_voltage_potential():
    fs = gp.fluid_system('CO, CO2, H2, O2, H2O, N2')
    fl1 = gp.fluid({'H2O': 0.5, 'H2': 0.5}, fs)
    fl2 = gp.fluid({'O2': 0.2, 'N2': 0.8}, fs)

    potential = mb.voltage_potential(fl1, fl2, 300, 1e5)  # V
    #assert potential == pytest.approx(1.1736986, abs=1e-4)

    potential = mb.voltage_potential(fl1, fl2, 273.15 + 800, 1e5)  # V
    assert potential == pytest.approx(0.9397330, abs=1e-4)

    fl1 = gp.fluid({'H2O': 0.01, 'H2': 0.99}, fs)
    fl2 = gp.fluid({'O2': 0.99, 'N2': 0.01}, fs)

    potential = mb.voltage_potential(fl1, fl2, 273.15 + 800, 1e5)  # V
    #assert potential == pytest.approx(1.18918092, abs=1e-4)

    fl1 = gp.fluid({'H2O': 0.90, 'H2': 0.10}, fs)
    fl2 = gp.fluid({'O2': 0.20, 'N2': 0.80}, fs)

    potential = mb.voltage_potential(fl1, fl2, 273.15 + 800, 1e5)  # V
    assert potential == pytest.approx(0.83813680, abs=1e-4)



def test_voltage_potential_without_o2():
    fl1 = gp.fluid({'H2O': 0.5, 'CO': 0, 'CO2': 0, 'H2': 0.5})
    fs = gp.fluid_system('CO, CO2, H2, O2, H2O, N2')
    fl2 = gp.fluid({'O2': 0.2, 'N2': 0.8}, fs)

    potential = mb.voltage_potential(fl1, fl2, 300, 1e5)  # V
    #assert potential == pytest.approx(1.1736986, abs=1e-4)

    potential = mb.voltage_potential(fl1, fl2, 273.15 + 800, 1e5)  # V
    #assert potential == pytest.approx(0.9397330, abs=1e-4)
"""
