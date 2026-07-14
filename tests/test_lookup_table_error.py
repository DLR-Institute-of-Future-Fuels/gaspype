"""
Testing the interpolation error of the lookup table for Cp, G, and H for multiple species.
"""

import gaspype as gp
import numpy as np
from gaspype.constants import R

fl = gp.fluid({'CH4': 1, 'CO2': 1, 'H2O': 1, 'SO2': 1, 'NH3': 1, 'NO2': 1, 'H2S': 1, 'C2H6': 1, 'C3H8': 1, 'H2': 1, 'O2': 1})
t_values = np.arange(200, 2000, step=10, dtype=int)


def test_cp_continuity_multi_species():
    cp_actual = fl.fs.get_species_cp(t_values)
    cp_neighbor_mean = (fl.fs.get_species_cp(t_values - 1) + fl.fs.get_species_cp(t_values + 1)) / 2
    diff = (cp_actual - cp_neighbor_mean) / cp_actual / 2
    avg_diff = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))
    print(f'Average difference of Cp: {avg_diff} J/mol/K')
    print(f'Max difference of Cp: {max_diff} J/mol/K')
    assert avg_diff < 1e-5
    assert max_diff < 1e-5


def test_g_continuity_multi_species():
    g_actual = fl.fs.get_species_g_rt(t_values) * R * np.reshape(t_values, (-1, 1))
    g_neighbor_mean = (fl.fs.get_species_g_rt(t_values - 1) + fl.fs.get_species_g_rt(t_values + 1)) / 2 * R * np.reshape(t_values, (-1, 1))

    diff = (g_actual - g_neighbor_mean) / g_actual / 2
    avg_diff = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))
    print(f'Average difference of G: {avg_diff} J/mol')
    print(f'Max difference of G: {max_diff} J/mol')
    assert avg_diff < 1e-4
    assert max_diff < 1e-4


def test_h_continuity_multi_species():
    h_actual = fl.fs.get_species_h(t_values)
    h_neighbor_mean = (fl.fs.get_species_h(t_values - 1) + fl.fs.get_species_h(t_values + 1)) / 2

    rel_diff = (h_actual - h_neighbor_mean) / h_actual / 2
    avg_diff = np.mean(np.abs(rel_diff))
    max_diff = np.max(np.abs(rel_diff))
    print(f'Average difference of H: {avg_diff} J/mol')
    print(f'Max difference of H: {max_diff} J/mol')
    assert avg_diff < 1e-4
    assert max_diff < 1e-4


if __name__ == "__main__":
    test_cp_continuity_multi_species()
    test_g_continuity_multi_species()
    test_h_continuity_multi_species()
