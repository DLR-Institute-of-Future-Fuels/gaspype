import cantera as ct
import numpy as np
import time
import gaspype as gp

gas = ct.Solution("gri30.yaml")
n_species = gas.n_species
n_states = 1_000_000

# Random temperatures and pressures
temperatures = np.linspace(300.0, 2500.0, n_states)
pressures = np.full(n_states, ct.one_atm)

# Generate random compositions for H2, H2O, N2
rng = np.random.default_rng(seed=42)
fractions = rng.random((n_states, 3))
fractions /= fractions.sum(axis=1)[:, None]  # normalize

# Convert to full 53-species mole fraction array
X = np.zeros((n_states, n_species))
X[:, gas.species_index('H2')] = fractions[:, 0]
X[:, gas.species_index('H2O')] = fractions[:, 1]
X[:, gas.species_index('N2')] = fractions[:, 2]

# Build SolutionArray
states = ct.SolutionArray(gas, n_states)

time.sleep(0.5)

# Vectorized assignment
t0 = time.perf_counter()
states.TPX = temperatures, pressures, X
cp_values = states.cp_mole
elapsed = time.perf_counter() - t0

print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (vectorized cantera)")
print("First 5 Cp values (J/mol-K):", cp_values[:5] / 1000)


# Vectorized fluid creation
fluid = (
    gp.fluid({'H2': 1}) * fractions[:, 0]
    + gp.fluid({'H2O': 1}) * fractions[:, 1]
    + gp.fluid({'N2': 1}) * fractions[:, 2]
)

time.sleep(0.5)

# Benchmark: calculate Cp for all states at once
t0 = time.perf_counter()
cp_values = fluid.get_cp(t=temperatures)
elapsed = time.perf_counter() - t0

print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (vectorized Gaspype)")
print("First 5 Cp values (J/mol·K):", cp_values[:5])