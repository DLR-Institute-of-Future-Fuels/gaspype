import gaspype as gp
import numpy as np

fs = gp.fluid_system('CO, CO2, H2, O2, H2O, N2')

fl = gp.fluid({'H2O': 0.99, 'H2': 0.01}, fs) * np.ones([2, 3, 4])
el = gp.elements(fl)


def test_str_index():
    assert fl['CO2'].shape == (2, 3, 4)
    assert el['C'].shape == (2, 3, 4)


def test_str_list_index():
    assert fl[['CO2', 'H2', 'CO']].shape == (2, 3, 4, 3)
    assert el[['C', 'H', 'O']].shape == (2, 3, 4, 3)


def test_int_list_index():
    assert fl[[1, 2, 0, 5]].shape == (2, 3, 4, 4)
    assert el[[1, 2, 0, 3]].shape == (2, 3, 4, 4)


def test_mixed_list_index():
    assert el[[1, 'H', 0, 'O']].shape == (2, 3, 4, 4)


def test_int_index():
    assert fl[5].shape == (2, 3, 4)
    assert el[-1].shape == (2, 3, 4)


def test_slice_index():
    assert fl[0:3].shape == (2, 3, 4, 3)
    assert fl[:].shape == (2, 3, 4, 6)

    assert el[0:3].shape == (2, 3, 4, 3)
    assert el[:].shape == (2, 3, 4, 4)
