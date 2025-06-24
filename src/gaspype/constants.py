"""
# Constants
This module contains physical constants used in the gas phase calculations.

kB: Boltzmann constant (J/K)
NA: Avogadro's number (1/mol)
R: Ideal gas constant (J/mol/K)
F: Faraday constant (C/mol)
p0: Standard pressure 1e5 Pa
t0: Standard temperature 298.15 K (25 °C)
p_atm: Standard atmosphere 1 atm = 101325 Pa
epsy: Small value for numerical stability (1e-18)
"""


kB = 1.380649e-23  # J/K
NA = 6.02214076e23  # 1/mol
R = kB * NA  # J/mol/K
F = 96485.3321233100  # C/mol
p0 = 1e5  # Pa
t0 = 273.15 + 25  # K
p_atm = 101325  # Pa

epsy = 1e-30
