import pytest
import gaspype as gp


def test_calculation_cold():
    # Testing a non-equilibrium case. Result is only
    # determined by stoichiometry.
    fs = gp.fluid_system(['CH4', 'H2O', 'H2', 'CO2', 'CO', 'O2'])

    el = gp.elements(gp.fluid({'H2': 1, 'CH4': 0.08, 'O2': 0.05}), fs=fs)

    composition = gp.equilibrium(el, 300, 1e5)

    print(el)
    print(composition)

    ref_result = [8.00000000e-02, 1.00000000e-01, 9.00000000e-01, 3.01246781e-23,
                  2.90783583e-27, 3.60487456e-82]

    assert composition.array_composition == pytest.approx(ref_result, abs=0.001)
