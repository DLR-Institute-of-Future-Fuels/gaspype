import gaspype as gp
import numpy as np
import pandas as pd


def test_fluid():
    fl = gp.fluid({'O2': 1, 'H2': 2, 'H2O': 3})

    df = pd.DataFrame(list(fl))
    assert df.shape == (1, 3)

    df = pd.DataFrame(list(fl * np.array([1, 2, 3, 4])))
    assert df.shape == (4, 3)


def test_elements():
    fl = gp.fluid({'O2': 1, 'H2': 2, 'H2O': 3})

    df = pd.DataFrame(list(gp.elements(fl)))
    assert df.shape == (1, 2)

    df = pd.DataFrame(list(gp.elements(fl * np.array([1, 2, 3, 4]))))
    assert df.shape == (4, 2)


def test_iter():
    fl = gp.fluid({'O2': 1, 'H2': 2, 'H2O': 3})

    fl2 = fl * np.array([1, 2, 3, 4])
    for i, f in enumerate(fl2):
        if i == 1:
            assert f == {'O2': np.float64(2.0), 'H2': np.float64(4.0), 'H2O': np.float64(6.0)}
        if i == 3:
            assert f == {'O2': np.float64(4.0), 'H2': np.float64(8.0), 'H2O': np.float64(12.0)}
