import gaspype as gp
import numpy as np
# import pytest
import cantera as ct  # type: ignore


def test_equilibrium_cantera():
    # Compare equilibrium calculations to Cantera results

    # gp.set_solver('system of equations')
    # gp.set_solver('gibs minimization')

    # fs = gp.fluid_system(['CH4', 'C2H6', 'C3H8', 'H2O', 'H2', 'CO2', 'CO', 'O2'])
    fs = gp.fluid_system(['CH4', 'H2O', 'H2', 'CO2', 'CO', 'O2'])
    # fs = gp.fluid_system([s for s in flow1.species_names if s in gps])

    composition = gp.fluid({'H2': 1}, fs) +\
        gp.fluid({'CH4': 1}, fs) * np.linspace(0, 0.05, 30) +\
        gp.fluid({'O2': 1}, fs) * np.expand_dims(np.linspace(0, 0.5, 30), axis=1)

    t = 1495 + 273.15  # K
    p = 1e5  # Pa

    fl = gp.equilibrium(composition, t, p)
    data = fl.get_x()
    gp_result_array = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))

    flow1 = ct.Solution('gri30.yaml')  # type: ignore
    ct_results = []
    comp = (composition.array_fractions[i, j] for i in range(composition.shape[0]) for j in range(composition.shape[1]))
    for c in comp:
        comp_dict = {s: v for s, v in zip(fs.species, c)}

        flow1.TP = t, p
        flow1.X = comp_dict
        flow1.equilibrate('TP')  # type: ignore
        indeces = [i for flsn in fs.active_species for i, sn in enumerate(flow1.species_names) if flsn == sn]  # type: ignore
        ct_results.append(flow1.X[indeces])  # type: ignore

        #if flow1.X[indeces][0] > 0.01:
        #    print(flow1.X[indeces])

    ct_result_array = np.stack(ct_results)  # type: ignore

    max_deviation = np.max(np.abs((gp_result_array - ct_result_array)))
    mean_deviation = np.mean(np.abs((gp_result_array - ct_result_array)))

    assert max_deviation < 0.04
    assert mean_deviation < 2e-4
