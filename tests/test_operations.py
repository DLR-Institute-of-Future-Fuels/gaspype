import gaspype as gp


def test_op_same_fluid_systems():
    fl1 = gp.fluid({'N2': 1, 'O2': 1})
    fl2 = gp.fluid({'O2': 1}, fl1.fs)

    fl_ret = fl1 + fl2

    assert isinstance(fl_ret, gp.fluid)

    assert fl1.fs is fl2.fs
    assert fl_ret.fs is fl1.fs

    fl_ret = fl1 - fl2

    assert fl1.fs is fl2.fs
    assert fl_ret.fs is fl1.fs


def test_op_different_fluid_systems():
    # no fs subset -> new fs
    fl1 = gp.fluid({'N2': 1, 'O2': 1})
    fl2 = gp.fluid({'NH3': 1, 'O2': 2})
    fl_ret = fl1 + fl2

    assert isinstance(fl_ret, gp.fluid)

    assert fl1.fs is not fl_ret.fs
    assert fl2.fs is not fl_ret.fs

    # fl1 is a subset of fl2 -> fl_ret uses fs of fl2
    fl1 = gp.fluid({'N2': 5, 'O2': 1})
    fl2 = gp.fluid({'N2': 11, 'O2': 12, 'NH3': 20})

    fl_ret = fl1 + fl2
    assert fl1.fs is not fl_ret.fs
    assert fl2.fs is fl_ret.fs

    fl_ret = fl2 - fl1
    assert fl1.fs is not fl_ret.fs
    assert fl2.fs is fl_ret.fs
    assert isinstance(fl_ret, gp.fluid)
    assert fl_ret.total == 37


def test_op_different_fluid_systems_elements():
    # no fs subset -> new fs
    fl1 = gp.fluid({'N2': 1, 'O2': 1})
    fl2 = gp.fluid({'NH3': 1, 'O2': 2})

    el_ret = gp.elements(fl1) + gp.elements(fl2)
    assert fl1.fs is not el_ret.fs
    assert fl2.fs is not el_ret.fs

    el_ret = gp.elements(fl1) - gp.elements(fl2)
    assert fl1.fs is not el_ret.fs
    assert fl2.fs is not el_ret.fs

    # fl1 is a subset of fl2 -> fl_ret uses fs of fl2
    fl1 = gp.fluid({'N2': 1, 'O2': 1})
    fl2 = gp.fluid({'N2': 1, 'O2': 1, 'NH3': 1})

    el_ret = gp.elements(fl1) + gp.elements(fl2)
    assert fl1.fs is not el_ret.fs
    assert fl2.fs is el_ret.fs

    el_ret = gp.elements(fl2) - gp.elements(fl1)
    assert fl1.fs is not el_ret.fs
    assert fl2.fs is el_ret.fs


def test_op_different_fluid_systems_mixed():
    # no fs subset -> new fs
    fl1 = gp.fluid({'N2': 1, 'O2': 1})
    fl2 = gp.fluid({'NH3': 1, 'O2': 2})

    el_ret = fl1 + gp.elements(fl2)

    assert isinstance(el_ret, gp.elements)

    assert fl1.fs is not el_ret.fs
    assert fl2.fs is not el_ret.fs

    el_ret = gp.elements(fl1) + fl2
    assert fl1.fs is not el_ret.fs
    assert fl2.fs is not el_ret.fs
    assert isinstance(el_ret, gp.elements)
    assert {'H', 'O', 'N'} == el_ret.get_elemental_composition().keys()

    # fl1 is a subset of fl2 -> fl_ret uses fs of fl2
    fl1 = gp.fluid({'N2': 1, 'O2': 1})
    fl2 = gp.fluid({'N2': 1, 'O2': 1, 'NH3': 1})

    el_ret = gp.elements(fl1) + fl2
    assert fl1.fs is not el_ret.fs
    assert fl2.fs is el_ret.fs

    el_ret = fl2 - gp.elements(fl1)
    assert fl1.fs is not el_ret.fs
    assert fl2.fs is el_ret.fs


def test_op_elements_same_fluid_systems():
    fs = gp.fluid_system('N2, O2')

    el1 = gp.elements({'N': 1, 'O': 1}, fs)
    el2 = gp.elements({'O': 1}, fs)

    el_ret = el1 + el2

    assert isinstance(el_ret, gp.elements)

    assert el1.fs is el2.fs
    assert el_ret.fs is el1.fs

    el_ret = el1 - el2

    assert el1.fs is el2.fs
    assert el_ret.fs is el1.fs


def test_op_elements_different_fluid_systems():
    fs1 = gp.fluid_system('N2, O2')
    fs2 = gp.fluid_system('O2')
    el1 = gp.elements({'N': 1, 'O': 1}, fs1)
    el2 = gp.elements({'O': 1}, fs2)

    el_ret = el1 + el2

    assert isinstance(el_ret, gp.elements)

    assert el1.fs is not el2.fs
    assert el2.fs is not el_ret.fs
    assert el1.fs is el_ret.fs

    el_ret = el1 - el2

    assert el1.fs is not el2.fs
    assert el2.fs is not el_ret.fs
    assert el1.fs is el_ret.fs


def test_op_tt():
    fl = gp.fluid({'H2O': 1, 'H2': 2})
    el = gp.elements({'N': 1, 'Cl': 2})

    assert isinstance(fl + el, gp.elements)

    assert 'H' in (fl + el).elements and \
           'Cl' in (fl + el).elements and \
           'O' in (fl + el).elements
