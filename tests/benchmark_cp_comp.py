import cantera as ct
import numpy as np
import time
import gaspype as gp
from benchmark_jax import jax_devices, to_device, measure

try:
    import jax
except ImportError:
    pass

try:
    import cea
    CEA_AVAILABLE = True
except ImportError:
    CEA_AVAILABLE = False

# Restrict Cantera to the same species as Gaspype for a fair comparison
gri30 = ct.Solution("gri30.yaml")
gas = ct.Solution(thermo='ideal-gas', species=[s for s in gri30.species() if s.name in ('H2', 'H2O', 'N2')])
n_species = gas.n_species
n_states = 1_000_000

# Random temperatures and pressures
temperatures = np.linspace(300.0, 2000.0, n_states)
pressures = np.full(n_states, ct.one_atm)

# Generate random compositions for H2, H2O, N2
rng = np.random.default_rng(seed=42)
fractions = rng.random((n_states, 3))
fractions /= fractions.sum(axis=1)[:, None]  # normalize

# Convert to mole fraction array in Cantera species order
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

for name, device in jax_devices():
    # The backend and the device are selected by the fluid system
    fs_jax = gp.fluid_system(fluid.fs.species, backend='jax', device=device)

    def get_cp(composition, t):
        return gp.fluid(composition, fs_jax).get_cp(t)

    composition_jax = to_device(fluid.array_composition, device)
    temperatures_jax = to_device(temperatures, device)

    first, elapsed, cp_values = measure(get_cp, composition_jax, temperatures_jax)
    print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (Gaspype with JAX on {name}, first call {first:.4f} seconds)")

    first, elapsed, cp_values = measure(jax.jit(get_cp), composition_jax, temperatures_jax)
    print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (Gaspype with JAX jit on {name}, first call {first:.4f} seconds)")
    print("First 5 Cp values (J/mol·K):", cp_values[:5])


if CEA_AVAILABLE:
    MW = np.array([2.016, 18.015, 28.014])
    mass_weights = fractions * MW / (fractions * MW).sum(axis=1)[:, None]
    avg_MW = np.sum(fractions * MW, axis=1)
    p_bar = cea.units.atm_to_bar(1.0)
    cea_mix = cea.Mixture(['H2', 'H2O', 'N2'])

    time.sleep(0.5)

    # the current NASA CEA Python API does not provide a NumPy-style
    # vectorized interface for thermodynamic property lookups
    t0 = time.perf_counter()
    cea_cp = np.zeros(n_states)
    for i in range(n_states):
        cea_cp[i] = cea_mix.calc_property(cea.FROZEN_CP, mass_weights[i], temperatures[i], p_bar)
    elapsed = time.perf_counter() - t0

    cea_cp_molar = cea_cp * avg_MW / 1000

    print(f"Computed {n_states} Cp values in {elapsed:.4f} seconds (CEA)")
    print("First 5 Cp values (J/mol-K):", cea_cp_molar[:5])
