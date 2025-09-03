import gaspype as gp
import numpy as np

fs = gp.fluid_system('CO, CO2, H2, O2, H2O, N2')

fl = gp.fluid({'H2O': 0.99, 'H2': 0.01}, fs) * np.ones([2, 3, 4])
el = gp.elements(fl)


def test_str_index():
    assert fl['CO2'].shape == (2, 3, 4)
    assert el['C'].shape == (2, 3, 4)


def test_single_axis_int_index():
    assert fl[0].shape == (3, 4)
    assert fl[1].shape == (3, 4)
    assert el[1].shape == (3, 4)
    assert el[0].shape == (3, 4)


def test_single_axis_int_list():
    assert fl[:, [0, 1]].shape == (2, 2, 4)
    assert el[:, [0, 1]].shape == (2, 2, 4)


def test_multi_axis_int_index():
    assert fl[0, 1].shape == (4,)
    assert fl[0, 1, 2].shape == tuple()
    assert fl[0, 2].shape == (4,)
    assert fl[:, 2, :].shape == (2, 4)
    assert fl[0, [1, 2]].shape == (2, 4)
    assert fl[..., 0].shape == (2, 3)
    assert el[0, 1].shape == (4,)
    assert el[0, 1, 2].shape == tuple()
    assert el[0, 2].shape == (4,)
    assert el[:, 2, :].shape == (2, 4)
    assert el[0, [1, 2]].shape == (2, 4)
    assert el[..., 0].shape == (2, 3)
