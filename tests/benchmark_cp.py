import cantera as ct
import numpy as np
import time
import gaspype as gp

gas = ct.Solution("gri30.yaml")
composition = {"H2": 0.3, "H2O": 0.3, "N2": 0.4}

n_species = gas.n_species
n_states = 1_000_000

# Random temperatures and pressures
temperatures = np.linspace(300.0, 2500.0, n_states)
pressures = np.full(n_states, ct.one_atm)

# Create a SolutionArray with many states at once
states = ct.SolutionArray(gas, len(temperatures))

time.sleep(0.5)

# Vectorized assignment
t0 = time.perf_counter()
states.TPX = temperatures, pressures, composition
cp_values = states.cp_mole
elapsed = time.perf_counter() - t0

print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (vectorized cantera)")
print("First 5 Cp values (J/mol-K):", cp_values[:5] / 1000)


# Vectorized fluid creation
fluid = gp.fluid(composition)

time.sleep(0.5)

# Benchmark: calculate Cp for all states at once
t0 = time.perf_counter()
cp_values = fluid.get_cp(t=temperatures)
elapsed = time.perf_counter() - t0

print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (vectorized Gaspype)")
print("First 5 Cp values (J/mol·K):", cp_values[:5])