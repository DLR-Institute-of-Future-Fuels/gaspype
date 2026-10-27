import gaspype as gp
import pytest
import warnings


def test_patter_filter():
    species_list = gp.species('H*O')
    assert 'H2O' in species_list


def test_regex_filter():
    species_list = gp.species('H.*O', use_regex=True)
    assert 'H2O' in species_list


def test_element_filter():
    species_list = gp.species(element_names='O, Cl')
    assert 'ClO2' in species_list

    species_list = gp.species(element_names=['O', 'Cl'])
    assert 'ClO2' in species_list and 'Cl2' in species_list and 'O2' in species_list


def test_combined_filter():
    species_list = gp.species('Cl*', 'O, Cl')
    assert 'ClO2' in species_list and 'Cl2' in species_list and 'O2' not in species_list


def test_fluid_system_temperature_range_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        gp.fluid_system('H2, CH4', t_max=6000)
        gp.species_system('H2O(L), CH4', t_max=6000)

    with pytest.warns(UserWarning, match=r'CH4 \(200 K to 6000 K\)') as record:
        gp.fluid_system('H2, CH4', t_max=7000)
    assert 'H2 ' not in str(record[0].message)
