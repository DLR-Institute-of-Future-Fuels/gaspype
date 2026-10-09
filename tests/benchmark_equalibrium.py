import cantera as ct
import gaspype as gp
import numpy as np
import time
from gaspype import fluid_system
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

# -----------------------
# Settings
# -----------------------

n_temps = 1000
temps_C = np.linspace(300, 1000, n_temps)     # °C
temperatures = temps_C + 273.15               # K
pressure = 101325                             # Pa (1 atm)

composition = {"CH4": 0.8, "H2O": 0.2}
species_to_track = ["CH4", "H2O", "CO", "CO2", "H2", "O2"]

# -----------------------
# Cantera benchmark
# -----------------------
# Restrict Cantera to the same species as Gaspype for a fair comparison
gri30 = ct.Solution("gri30.yaml")
gas = ct.Solution(thermo='ideal-gas', species=[s for s in gri30.species() if s.name in species_to_track])

states = ct.SolutionArray(gas, n_temps)
states.TPX = temperatures, pressure, composition

time.sleep(0.5)
t0 = time.perf_counter()
states.equilibrate('TP')
elapsed_cantera = time.perf_counter() - t0
print(f"Cantera: {elapsed_cantera:.4f} s")

eq_cantera = states.X[:, [gas.species_index(s) for s in species_to_track]]

# -----------------------
# Gaspype benchmark
# -----------------------
# Construct the fluid with composition and tracked species
fluid = gp.fluid(composition, fs=fluid_system(species_to_track))

time.sleep(0.5)
t0 = time.perf_counter()
eq_gaspype = gp.equilibrium(fluid, t=temperatures, p=pressure)
elapsed_gaspype = time.perf_counter() - t0
print(f"Gaspype: {elapsed_gaspype:.4f} s")

# Check if elemental balance of result is correct
el_err = np.sum((gp.elements(eq_gaspype) - gp.elements(fluid)).get_n()**2)
assert np.all(el_err < 1e-20)

if CEA_AVAILABLE:
    reac_names = ["CH4", "H2O"]
    prod_names = ["CH4", "H2O", "CO", "CO2", "H2", "O2", "H", "O", "OH"]

    reac = cea.Mixture(reac_names)
    prod = cea.Mixture(prod_names)

    solver = cea.EqSolver(prod, reactants=reac)
    solution = cea.EqSolution(solver)

    input_moles = np.array([8.0, 2.0])
    input_weights = reac.moles_to_weights(input_moles)

    p_bar = pressure * 1e-5  # Convert to bar

    eq_cea = np.zeros((n_temps, len(species_to_track)))

    time.sleep(0.5)
    t0 = time.perf_counter()
    for i, T in enumerate(temperatures):
        solver.solve(solution, cea.TP, T, p_bar, input_weights)
        if solution.converged:
            for j, s in enumerate(species_to_track):
                eq_cea[i, j] = solution.mole_fractions.get(s, 0.0)
    elapsed_cea = time.perf_counter() - t0
    print(f"CEA: {elapsed_cea:.4f} s")

# -----------------------
# Compare first 5 results
# -----------------------
print("\nFirst 5 equilibrium compositions:")
for i in range(5):
    print(f"T = {temperatures[i]:.1f} K")
    print("  Cantera:", eq_cantera[i])
    print("  Gaspype :", eq_gaspype.array_composition[i])
    if CEA_AVAILABLE:
        print("  CEA     :", eq_cea[i])

# -----------------------
# Gaspype with NumPy and JAX for larger numbers of states
# -----------------------
devices = jax_devices()

print("\nGaspype equilibrium run time for increasing numbers of states:")
for n_states in [1_000, 100_000, 1_000_000]:
    temperatures = np.linspace(300, 1000, n_states) + 273.15

    t0 = time.perf_counter()
    eq_numpy = gp.equilibrium(fluid, t=temperatures, p=pressure).array_composition
    elapsed_numpy = time.perf_counter() - t0
    print(f"{n_states:9} states, NumPy:          {elapsed_numpy:8.4f} s")

    for name, device in devices:
        # The backend and the device are selected by the fluid system
        fluid_jax = gp.fluid(composition, fs=fluid_system(species_to_track, backend='jax', device=device))
        temperatures_jax = to_device(temperatures, device)

        def equilibrium_composition(t):
            return gp.equilibrium(fluid_jax, t=t, p=pressure).array_composition

        for label, func in [(f"JAX on {name}:    ", equilibrium_composition),
                            (f"JAX jit on {name}:", jax.jit(equilibrium_composition))]:
            first, elapsed_jax, eq_jax = measure(func, temperatures_jax, repeats=2)
            assert np.allclose(eq_jax, eq_numpy, rtol=1e-6, atol=1e-12)
            print(f"{n_states:9} states, {label} {elapsed_jax:8.4f} s (first call {first:.4f} s, "
                  f"{elapsed_numpy / elapsed_jax:.1f} x the speed of NumPy)")
