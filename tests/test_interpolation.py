import gaspype as gp
import numpy as np
import pytest
from gaspype._main import species_properties, _species_db

fs = gp.fluid_system('CH4, H2O, H2, CO2, CO')

# Off-grid temperatures, the NASA9 polynomials are discontinuous at 1000 K
t = np.concatenate([np.linspace(300.3, 998.7, 777), np.linspace(1001.3, 1990.7, 777)])
cp_ref, h_ref, s_ref, g_rt_ref = (np.array(p).T for p in zip(*(species_properties(_species_db.read(s), t)
                                                               for s in fs.species)))


def test_interpolation_accuracy():
    assert fs.get_species_h(t) == pytest.approx(h_ref, rel=0, abs=1e-6)
    assert fs.get_species_s(t) == pytest.approx(s_ref, rel=0, abs=1e-8)
    assert fs.get_species_g_rt(t) == pytest.approx(g_rt_ref, rel=0, abs=1e-8)
    assert fs.get_species_cp(t) == pytest.approx(cp_ref, rel=0, abs=1e-2)


def test_interpolation_derivatives():
    # Derivatives of the interpolated properties are continuous at the supporting
    # points and thermodynamically consistent
    dt = 1e-4
    for node in [500.0, 1500.0]:
        for tc in [node - 0.5 * dt, node + 0.5 * dt]:
            dh = (fs.get_species_h(tc + 0.5 * dt) - fs.get_species_h(tc - 0.5 * dt)) / dt
            ds = (fs.get_species_s(tc + 0.5 * dt) - fs.get_species_s(tc - 0.5 * dt)) / dt
            dg_rt = (fs.get_species_g_rt(tc + 0.5 * dt) - fs.get_species_g_rt(tc - 0.5 * dt)) / dt
            assert dh == pytest.approx(fs.get_species_cp(node), rel=1e-6)
            assert ds == pytest.approx(fs.get_species_cp(node) / node, rel=1e-6)
            assert dg_rt == pytest.approx(-fs.get_species_h(node) / (gp.constants.R * node ** 2), rel=1e-5)


def test_extrapolation():
    # Outside of the table the values are extrapolated linearly with the boundary slope
    t_min, t_max = 250, 1999
    for t_out, t_edge in [(200.0, t_min), (2100.0, t_max)]:
        h = fs.get_species_h(t_out)
        assert h == pytest.approx(fs.get_species_h(t_edge) + (t_out - t_edge) * fs.get_species_cp(t_edge), rel=1e-12)
