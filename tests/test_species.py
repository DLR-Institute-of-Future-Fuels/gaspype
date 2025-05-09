import gaspype as gp


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
