import gaspype as gp


def test_set_and_read_solver():
    tmp = gp.get_solver()
    gp.set_solver('gibs minimization')
    assert gp.get_solver() == 'gibs minimization'
    gp.set_solver(tmp)


def test_fluid_stacking_concat():
    fl1 = gp.fluid({'O2': 1, 'N2': 0})
    fl2 = gp.fluid({'N2': 1}, fl1.fs)

    fl3 = gp.stack([fl1, fl2])
    assert fl3.shape == (2,)

    fl4 = gp.stack([fl3, fl3])
    assert fl4.shape == (2, 2)

    fl5 = gp.concat([fl3, fl3])
    assert fl5.shape == (4,)


def test_elements_stacking_concat():
    el1 = gp.elements(gp.fluid({'O2': 1, 'N2': 0}))
    el2 = gp.elements(gp.fluid({'N2': 1}), el1.fs)

    assert el1.fs == el2.fs

    el3 = gp.stack([el1, el2])
    assert el3.shape == (2,)

    el4 = gp.stack([el3, el3])
    assert el4.shape == (2, 2)

    el5 = gp.concat([el3, el3])
    assert el5.shape == (4,)


def test_element_casting():
    fl1 = gp.fluid({'O2': 1, 'N2': 2, 'H2': 3})

    el1 = gp.elements(fl1)
    assert el1.get_elemental_composition() == {'N': 4.0, 'H': 6.0, 'O': 2.0}

    fs = gp.fluid_system('H2, O2, N2')
    el2 = gp.elements(fl1, fs)
    assert el2.get_elemental_composition() == {'N': 4.0, 'H': 6.0, 'O': 2.0}
