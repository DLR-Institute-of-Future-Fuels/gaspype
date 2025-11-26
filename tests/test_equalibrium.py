import gaspype as gp


def test_single_equilibrium():
    # Compare equilibrium calculations to Cantera results

    # gp.set_solver('system of equations')
    # gp.set_solver('gibs minimization')

    # fs = gp.fluid_system(['CH4', 'C2H6', 'C3H8', 'H2O', 'H2', 'CO2', 'CO', 'O2'])
    fs = gp.fluid_system(['CH4', 'H2O', 'H2', 'CO2', 'CO', 'O2'])
    # fs = gp.fluid_system([s for s in flow1.species_names if s in gps])

    composition = gp.elements({'H': 2, 'O': 0, 'C': 0}, fs)

    t = 1495 + 273.15  # K
    p = 1e5  # Pa

    fl = gp.equilibrium(composition, t, p)

    print(fl)


if __name__ == "__main__":
    test_single_equilibrium()
