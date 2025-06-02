from gaspype._phys_data import db_reader


def test_db_reader():
    with open('src/gaspype/data/therm_data.bin', 'rb') as f:
        db = db_reader(f.read())

    assert 'HCl' in db
    assert 'TEST' not in db
    assert db['HCl'].name == 'HCl'
    assert db['CH4'].composition == {'C': 1, 'H': 4}
    assert db['H2O'].model == 9

    for species in db:
        print(species.name)
        assert species.model == 9
        assert len(species.name) > 0
        assert len(species.composition) > 0
        assert any(el in species.name for el in species.composition.keys())
