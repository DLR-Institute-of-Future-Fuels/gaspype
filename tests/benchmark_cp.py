import cantera as ct
import numpy as np
import time
import gaspype as gp

try:
    import cea
    CEA_AVAILABLE = True
except ImportError:
    CEA_AVAILABLE = False

gas = ct.Solution("gri30.yaml")
composition = {"H2": 0.3, "H2O": 0.3, "N2": 0.4}

n_species = gas.n_species
n_states = 1_000_000

temperatures = np.linspace(300.0, 2500.0, n_states)
pressures = np.full(n_states, ct.one_atm)

states = ct.SolutionArray(gas, len(temperatures))

time.sleep(0.5)

t0 = time.perf_counter()
states.TPX = temperatures, pressures, composition
cp_values = states.cp_mole
elapsed = time.perf_counter() - t0

print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (vectorized cantera)")
print("First 5 Cp values (J/mol-K):", cp_values[:5] / 1000)

fluid = gp.fluid(composition)

time.sleep(0.5)

t0 = time.perf_counter()
cp_values = fluid.get_cp(t=temperatures)
elapsed = time.perf_counter() - t0

print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (vectorized Gaspype)")
print("First 5 Cp values (J/mol·K):", cp_values[:5])

if CEA_AVAILABLE:
    cea_mix = cea.Mixture(['H2', 'H2O', 'N2'])
    mole_fracs = np.array([0.3, 0.3, 0.4])
    MW = np.array([2.016, 18.015, 28.014])
    mass_weights = mole_fracs * MW / (mole_fracs * MW).sum()
    avg_MW = np.sum(mole_fracs * MW)
    p_bar = cea.units.atm_to_bar(1.0)

    time.sleep(0.5)

    # the current NASA CEA Python API does not provide a NumPy-style
    # vectorized interface for thermodynamic property lookups
    t0 = time.perf_counter()
    cea_cp = np.zeros(n_states)
    for i in range(n_states):
        cea_cp[i] = cea_mix.calc_property(cea.FROZEN_CP, mass_weights, temperatures[i], p_bar)
    elapsed = time.perf_counter() - t0

    cea_cp_molar = cea_cp * avg_MW / 1000

    print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (CEA)")
    print("First 5 Cp values (J/mol-K):", cea_cp_molar[:5])
