import gaspype as gp


def test_one_component_equalibrium():
    fl = gp.fluid({'O2': 1})

    t = 858
    p = 2.56e5

    fl_eq = gp.equilibrium(fl, t, p)

    assert (fl.array_composition == fl_eq.array_composition).all()


def test_undipendent_components():
    fl = gp.fluid({'O2': 1, 'N2': 1, 'CH4': 1})

    t = 858
    p = 2.56e5

    fl_eq = gp.equilibrium(fl, t, p)

    assert (fl.array_composition == fl_eq.array_composition).all()


def test_restricted_component_equalibrium():
    fl = gp.fluid({'O2': 1, 'N2': 1, 'H2O': 1})

    t = 858
    p = 2.56e5

    fl_eq = gp.equilibrium(fl, t, p)

    assert (fl.array_composition == fl_eq.array_composition).all()


def test_comp_cache():

    t = 2500
    p = 1e5

    # Cached case 1:
    fl = gp.fluid({'H2': 1, 'CH4': 1})
    fl_2 = gp.equilibrium(fl, t, p)
    assert fl is fl_2

    # Cached case 2:
    fl = gp.fluid({'H2': 1, 'CH4': 1, 'O2': 10, 'N2': 1})
    fl_2 = gp.equilibrium(fl, t, p)
    assert fl is fl_2

    # Non cached case 2:
    fl = gp.fluid({'H2': 1, 'CO': 1, 'CO2': 1, 'CH4': 1})
    fl_3 = gp.equilibrium(fl, t, p)
    print(fl_3)
    print(fl is fl_3)
    assert not (fl.array_composition == fl_3.array_composition).all()

    # Non cached case 3:
    fl = gp.fluid({'O2': 1, 'CO': 1, 'CH4': 1, 'H2': 1})
    fl_3 = gp.equilibrium(fl, t, p)
    print(fl_3)
    print(fl is fl_3)
    assert not (fl.array_composition == fl_3.array_composition).all()
