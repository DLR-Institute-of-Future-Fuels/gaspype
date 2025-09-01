import cantera as ct
import gaspype as gp
import numpy as np
import time
from gaspype import fluid_system

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
gas = ct.Solution("gri30.yaml")
gas.TPX = temperatures[0], pressure, composition

eq_cantera = np.zeros((n_temps, len(species_to_track)))

time.sleep(0.5)
t0 = time.perf_counter()
for i, T in enumerate(temperatures):
    gas.TP = T, pressure
    gas.equilibrate('TP')
    eq_cantera[i, :] = [gas.X[gas.species_index(s)] for s in species_to_track]
elapsed_cantera = time.perf_counter() - t0
print(f"Cantera: {elapsed_cantera:.4f} s")

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

# -----------------------
# Compare first 5 results
# -----------------------
print("First 5 equilibrium compositions (mole fractions):")
for i in range(5):
    print(f"T = {temperatures[i]:.1f} K")
    print("  Cantera:", eq_cantera[i])
    print("  Gaspype :", eq_gaspype.array_composition[i])
