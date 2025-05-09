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
