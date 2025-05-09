import gaspype as gp
import pytest
import numpy as np


def test_elements_from_dict():
    el = gp.elements({'Si': 1, 'H': 1, 'O': 1})
    fl = gp.equilibrium(el, t=1000, p=1e5)

    assert 'SiO' in el.fs.species and \
           'SiH' in el.fs.species and \
           'H2' in el.fs.species

    assert fl.get_x('H2') == pytest.approx(0.3333, abs=0.5)


def test_elements_from_dict_with_fs():
    fs = gp.fluid_system('Si, O2, H2')
    el = gp.elements({'Si': 1, 'O': 2}, fs)
    fl = gp.equilibrium(el, t=1000, p=1e5)

    assert 'SiO' not in el.fs.species and \
           'Si' in el.fs.species and \
           'O2' in el.fs.species

    assert len(fl.fs.species) == 3
    assert np.sum(el) == 3


def test_elements_from_fluid():
    fl = gp.fluid({'SiO': 1, 'H2': 1, 'O2': 1})
    el = gp.elements(fl)

    assert 'SiO' in el.fs.species and \
           'SiH' not in el.fs.species and \
           'O2' in el.fs.species and \
           'H2' in el.fs.species


def test_elements_mass():
    el = gp.elements({'Si': 1, 'O': 2})
    assert el.get_mass() == pytest.approx(0.0601, abs=0.0001)
