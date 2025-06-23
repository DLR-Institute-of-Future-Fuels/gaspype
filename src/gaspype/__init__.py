"""
Gaaspype is a library for thermodynamic calculations

Main classes:
    - fluid: Represents a fluid.
    - elements: Represents chemical elements.
    - fluid_system: Represents a system of fluids.

Example usage:
    >>> import gaspype as gp
    >>> fl = gp.fluid({'H2O': 1, 'H2': 2})
    >>> cp = fl.get_cp(t=800+273.15)
"""

from ._main import species, fluid_system, fluid, elements
from ._operations import stack, concat, carbon_activity, oxygen_partial_pressure
from ._solver import set_solver, get_solver, equilibrium

__all__ = [
    'species', 'fluid_system', 'fluid', 'elements',
    'set_solver', 'get_solver', 'equilibrium',
    'stack', 'concat', 'carbon_activity', 'oxygen_partial_pressure']
