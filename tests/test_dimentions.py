import gaspype as gp
import numpy as np

fs = gp.fluid_system(gp.species('C2H#'))


def test_broadcast_temperature():
    fl = gp.fluid({'C2H4': 0.5, 'C2H6': 0.5}, fs)

    assert fl.shape == tuple()

    assert fl.array_fractions.shape == (len(fs.species),)

    s = fl.get_s(np.array([300, 400, 500, 600]), 1e6)
    assert s.shape == (4,)
    s = fl.get_g(np.array([300, 400, 500, 600]), 1e6)
    assert s.shape == (4,)
    s = fl.get_g_rt(np.array([300, 400, 500, 600]), 1e6)
    assert s.shape == (4,)
    s = fl.get_h(np.array([300, 400, 500, 600]))
    assert s.shape == (4,)
    s = fl.get_cp(np.array([300, 400, 500, 600]))
    assert s.shape == (4,)
    s = fl.get_v(np.array([300, 400, 500, 600]), 1e6)
    assert s.shape == (4,)
    s = fl.get_density(np.array([300, 400, 500, 600]), 1e6)
    assert s.shape == (4,)

    s = fl.get_s(np.array([[300, 400, 500, 600], [305, 405, 505, 605]]), 1e6)
    assert s.shape == (2, 4)
    s = fl.get_g(np.array([[300, 400, 500, 600], [305, 405, 505, 605]]), 1e6)
    assert s.shape == (2, 4)
    s = fl.get_g_rt(np.array([[300, 400, 500, 600], [305, 405, 505, 605]]), 1e6)
    assert s.shape == (2, 4)
    s = fl.get_h(np.array([[300, 400, 500, 600], [305, 405, 505, 605]]))
    assert s.shape == (2, 4)
    s = fl.get_cp(np.array([[300, 400, 500, 600], [305, 405, 505, 605]]))
    assert s.shape == (2, 4)
    s = fl.get_v(np.array([[300, 400, 500, 600], [305, 405, 505, 605]]), 1e6)
    assert s.shape == (2, 4)
    s = fl.get_density(np.array([[300, 400, 500, 600], [305, 405, 505, 605]]), 1e6)
    assert s.shape == (2, 4)

    s = fl.get_s(np.array([[300], [305]]), 1e6)
    assert s.shape == (2, 1)
    s = fl.get_g(np.array([[300], [305]]), 1e6)
    assert s.shape == (2, 1)
    s = fl.get_g_rt(np.array([[300], [305]]), 1e6)
    assert s.shape == (2, 1)
    s = fl.get_h(np.array([[300], [305]]))
    assert s.shape == (2, 1)
    s = fl.get_cp(np.array([[300], [305]]))
    assert s.shape == (2, 1)
    s = fl.get_v(np.array([[300], [305]]), 1e6)
    assert s.shape == (2, 1)
    s = fl.get_density(np.array([[300], [305]]), 1e6)
    assert s.shape == (2, 1)


def test_broadcast_temperature_ex():
    fl = gp.fluid({'C2H4': 0.5, 'C2H6': 0.5}, fs, [2, 4])

    assert fl.shape == (2, 4)

    assert fl.array_fractions.shape == (2, 4, len(fs.species))

    s = fl.get_s(np.array([300, 400, 500, 600]), 1e6)
    assert s.shape == (2, 4)

    s = fl.get_s(np.array([[300, 400, 500, 600], [305, 405, 505, 605]]), 1e6)
    assert s.shape == (2, 4)

    s = fl.get_s(np.array([[300], [305]]), 1e6)
    assert s.shape == (2, 4)


def test_broadcast_pressure():
    fl = gp.fluid({'C2H4': 0.5, 'C2H6': 0.5}, fs)

    assert fl.shape == tuple()

    assert fl.array_fractions.shape == (len(fs.species),)

    s = fl.get_s(800, np.array([1e5, 2e5, 3e5, 4e5]))
    assert s.shape == (4,)
    s = fl.get_g(800, np.array([1e5, 2e5, 3e5, 4e5]))
    assert s.shape == (4,)
    s = fl.get_g_rt(800, np.array([1e5, 2e5, 3e5, 4e5]))
    assert s.shape == (4,)
    s = fl.get_v(800, np.array([1e5, 2e5, 3e5, 4e5]))
    assert s.shape == (4,)
    s = fl.get_density(800, np.array([1e5, 2e5, 3e5, 4e5]))
    assert s.shape == (4,)

    s = fl.get_s(800, np.array([[1e5, 2e5, 3e5, 4e5], [1.5e5, 2.5e5, 3.5e5, 4.5e5]]))
    assert s.shape == (2, 4)
    s = fl.get_g(800, np.array([[1e5, 2e5, 3e5, 4e5], [1.5e5, 2.5e5, 3.5e5, 4.5e5]]))
    assert s.shape == (2, 4)
    s = fl.get_g_rt(800, np.array([[1e5, 2e5, 3e5, 4e5], [1.5e5, 2.5e5, 3.5e5, 4.5e5]]))
    assert s.shape == (2, 4)
    s = fl.get_v(800, np.array([[1e5, 2e5, 3e5, 4e5], [1.5e5, 2.5e5, 3.5e5, 4.5e5]]))
    assert s.shape == (2, 4)
    s = fl.get_density(800, np.array([[1e5, 2e5, 3e5, 4e5], [1.5e5, 2.5e5, 3.5e5, 4.5e5]]))
    assert s.shape == (2, 4)

    s = fl.get_s(800, np.array([[1e5], [1.5e5]]))
    assert s.shape == (2, 1)
    s = fl.get_g(800, np.array([[1e5], [1.5e5]]))
    assert s.shape == (2, 1)
    s = fl.get_g_rt(800, np.array([[1e5], [1.5e5]]))
    assert s.shape == (2, 1)
    s = fl.get_v(800, np.array([[1e5], [1.5e5]]))
    assert s.shape == (2, 1)
    s = fl.get_density(800, np.array([[1e5], [1.5e5]]))
    assert s.shape == (2, 1)


def test_equilibrium_on_temperature_range():
    fl = gp.fluid({'C2H4': 0.5, 'C2H6': 0.5}, fs)
    fl2 = gp.equilibrium(fl, np.linspace(300, 1000, 3), 1e5)

    assert fl2.shape == (3,)
