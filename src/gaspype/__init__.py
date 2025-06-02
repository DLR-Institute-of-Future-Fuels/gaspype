import numpy as np
from numpy.typing import NDArray
from typing import Literal, Sequence, Callable, Any, TypeVar, Iterator, overload
from math import log as ln, ceil, exp
from scipy.optimize import minimize, root
from scipy.linalg import null_space
from gaspype._phys_data import atomic_weights, db_reader
import re
import pkgutil

_Shape = tuple[int, ...]
NDFloat = np.float64
FloatArray = NDArray[NDFloat]

_T = TypeVar('_T', 'fluid', 'elements')

_data = pkgutil.get_data(__name__, 'data/therm_data.bin')
assert _data is not None, 'Could not load thermodynamic data'
_species_db = db_reader(_data)

kB = 1.380649e-23  # J/K
NA = 6.02214076e23  # 1/mol
R = kB * NA  # J/mol/K
F = 96485.3321233100  # C/mol
p0 = 1e5  # Pa
t0 = 273.15 + 25  # K
p_atm = 101325  # Pa

_epsy = 1e-18


def lookup(prop_array: FloatArray,
           temperature: FloatArray | float,
           t_offset: float) -> FloatArray:
    """linear interpolates values from the given prop_array

    Args:
        prop_array: Array of the temperature depended property
        temperature: Absolute temperature(s) in Kelvin. Must
            be broadcastable to prop_array.

    Returns:
        Interpolates values based on given temperature
    """
    t = np.array(temperature) - t_offset
    t_lim = np.minimum(np.maximum(0, t), prop_array.shape[0] - 2)

    f = np.expand_dims(t - np.floor(t_lim), axis=-1)

    ti1 = t_lim.astype(int)
    return f * prop_array[ti1 + 1, :] + (1 - f) * prop_array[ti1, :]


def species(pattern: str = '*', element_names: str | list[str] = [], use_regex: bool = False) -> list[str]:
    """Returns a alphabetically sorted list of all available species
    filtered by a pattern if supplied

    Args:
        pattern: Optional filter for specific molecules
            Placeholder characters:
                # A number including non written ones: 'C#H#' matches 'CH4'
                $ Arbitrary element name
                * Any sequence of characters
        element_names:
            restrict results to species that contain only the specified elements.
            The elements can be supplied as list of strings or as comma separated string.
        use_regex: using regular expression for the pattern

    Returns:
        List of species
    """
    if isinstance(element_names, str):
        elements = {s.strip() for s in element_names.split(',')}
    else:
        assert isinstance(element_names, list), 'type of element_names must be list or str'
        elements = set(element_names)

    for el in elements:
        assert el in atomic_weights, f'element {el} unknown'

    if not use_regex:
        el_pattern = '|'.join([el for el in atomic_weights.keys()])
        pattern = pattern.replace('*', '.*')
        pattern = pattern.replace('#', '\\d*')
        pattern = pattern.replace('$', '(' + el_pattern + ')')
        pattern = '^' + pattern + '(,.*)?$'

    if element_names == []:
        return [sn for sn in _species_db.names if re.fullmatch(pattern, sn)]
    else:
        return [
            s.name for s in _species_db
            if re.fullmatch(pattern, s.name) and
            (len(elements) == 0 or set(s.composition.keys()).issubset(elements))]


def set_solver(solver: Literal['gibs minimization', 'system of equations']) -> None:
    """Select a solver for chemical equilibrium.

    Args:
        solver: Name of the solver
    """
    global _equilibrium_solver
    if solver == 'gibs minimization':
        _equilibrium_solver = _equilibrium_gmin
    elif solver == 'system of equations':
        _equilibrium_solver = _equilibrium_eq
    else:
        raise ValueError('Unknown solver')


def get_solver() -> Literal['gibs minimization', 'system of equations']:
    """Returns the selected solver name.

    Returns:
        Solver name
    """
    if _equilibrium_solver == _equilibrium_gmin:
        return 'gibs minimization'
    else:
        assert _equilibrium_solver == _equilibrium_eq
        return 'system of equations'


class fluid_system:
    """A class to represent a fluid_system defined by a set of selected species.

    Attributes:
        species_names (list[str]): List of selected species in the fluid_system
        array_molar_mass: Array of the molar masses of the species in the fluid_system
        array_element_composition: Array of the element composition of the species in the fluid_system.
            Dimension is: (number of species, number of elements)
        array_atomic_mass: Array of the atomic masses of the elements in the fluid_system
    """

    def __init__(self, species: list[str] | str, t_min: int = 250, t_max: int = 2000):
        """Instantiates a fluid_system.

        Args:
            species: List of species names to be available in the constructed
                fluid_system (as list of strings or a comma separated string)
            t_min: Lower bound of the required temperature range in Kelvin
            t_max: Upper bound of the required temperature range in Kelvin
        """
        if isinstance(species, str):
            species = [s.strip() for s in species.split(',')]

        temperature_base_points = range(int(t_min), ceil(t_max))

        data_shape = (len(temperature_base_points), len(species))
        self._cp_array = np.zeros(data_shape)
        self._h_array = np.zeros(data_shape)
        self._s_array = np.zeros(data_shape)
        # self._g_array = np.zeros(data_shape)
        self._g_rt_array = np.zeros(data_shape)

        self._t_offset = int(t_min)
        self.species = species
        self.active_species = species
        element_compositions: list[dict[str, int]] = list()

        for i, s in enumerate(species):
            species_data = _species_db.read(s)
            if not species_data:
                raise Exception(f'Species {s} not found')
            element_compositions.append(species_data.composition)

            assert species_data.model == 9, 'Only NASA9 polynomials are supported'

            for t1, t2, a in zip(species_data.t_range[:-1], species_data.t_range[1:], species_data.data):

                for j, T in enumerate(temperature_base_points):
                    if t2 >= T >= t1:
                        self._cp_array[j, i] = R * (a[0]*T**-2 + a[1]*T**-1 + a[2] + a[3]*T
                                                    + a[4]*T**2 + a[5]*T**3 + a[6]*T**4)
                        self._h_array[j, i] = R*T * (-a[0]*T**-2 + a[1]*ln(T)/T + a[2]
                                                     + a[3]/2*T + a[4]/3*T**2 + a[5]/4*T**3
                                                     + a[6]/5*T**4 + a[7]/T)
                        self._s_array[j, i] = R * (-a[0]/2*T**-2 - a[1]*T**-1 + a[2]*ln(T)
                                                   + a[3]*T + a[4]/2*T**2 + a[5]/3*T**3
                                                   + a[6]/4*T**4 + a[8])
                        #self._g_array[j, i] = self._h_array[j, i] - self._s_array[j, i] * T
                        self._g_rt_array[j, i] = (self._h_array[j, i] / T - self._s_array[j, i]) / R

            # TODO: Check if temperature range is not available
            # print(f'Warning: temperature ({T}) out of range for {s}')

        self.elements: list[str] = sorted(list(set(k for ac in element_compositions for k in ac.keys())))
        self.array_species_elements = np.array([[ec[el] if el in ec else 0.0 for el in self.elements] for ec in element_compositions])

        self.array_atomic_mass = np.array([atomic_weights[el] for el in self.elements]) * 1e-3  # kg/mol
        self.array_molar_mass: FloatArray = np.sum(self.array_atomic_mass * self.array_species_elements, axis=-1)  # kg/mol

        self.array_stoichiometric_coefficients: FloatArray = np.array(null_space(self.array_species_elements.T), dtype=NDFloat).T

    def get_species_h(self, t: float | FloatArray) -> FloatArray:
        """Get the molar enthalpies for all species in the fluid system

        Args:
            t: Temperature in Kelvin (can be an array)

        Returns:
            Array with the enthalpies of each specie in J/mol
        """
        return lookup(self._h_array, t, self._t_offset)

    def get_species_s(self, t: float | FloatArray) -> FloatArray:
        """Get the molar entropies for all species in the fluid system

        Args:
            t: Temperature in Kelvin (can be an array)

        Returns:
            Array with the entropies of each specie in J/mol/K
        """
        return lookup(self._s_array, t, self._t_offset)

    def get_species_cp(self, t: float | FloatArray) -> FloatArray:
        """Get the isobaric molar heat capacity for all species in the fluid system

        Args:
            t: Temperature in Kelvin (can be an array)

        Returns:
            Array with the heat capacities of each specie in J/mol/K
        """
        return lookup(self._cp_array, t, self._t_offset)

    # def get_species_g(self, t: float | NDArray[_Float]) -> NDArray[_Float]:
    #     return lookup(self._g_array, t, self._t_offset)

    def get_species_g_rt(self, t: float | FloatArray) -> FloatArray:
        """Get specific gibbs free energy divided by RT for all species in the
        fluid system (g/R/T == (h/T-s)/R )

        Args:
            t: Temperature in Kelvin (can be an array)

        Returns:
            Array of gibbs free energy divided by RT (dimensionless)
        """
        return lookup(self._g_rt_array, t, self._t_offset)

    def get_species_references(self) -> str:
        """Get a string with the references for all fluids of the fluid system

        Returns:
            String with the references
        """
        return '\n'.join([f'{s:<12}: {_species_db[s].ref_string}' for s in self.species])

    def __add__(self, other: 'fluid_system') -> 'fluid_system':
        assert isinstance(other, self.__class__)
        return self.__class__(self.species + other.species)

    def __repr__(self) -> str:
        return ('Fluid system\n    Species:  ' + ', '.join(self.species) +
                '\n    Elements: ' + ', '.join(self.elements))


class fluid:
    """A class to represent a fluid defined by a composition of
    one or more species.

    Attributes:
        fs: Reference to the fluid_system used for this fluid
        species: List of species names in the associated fluid_system
        array_composition: Array of the molar amounts of the species in the fluid
        array_element_composition: Array of the element composition in the fluid
        array_fractions: Array of the molar fractions of the species in the fluid
        total: Array of the sums of the molar amount of all species
        fs: Reference to the fluid_system used for this fluid
    """

    __array_priority__ = 100

    def __init__(self, composition: dict[str, float] | list[float] | FloatArray,
                 fs: fluid_system | None = None,
                 shape: Sequence[int] | None = None):
        """Instantiates a fluid.

        Args:
            composition: A dict of species names with their composition, e.g.:
                {'O2':0.5,'H2O':0.5} or a list/numpy-array of compositions.
                The array can be multidimensional, the size of the last dimension
                must match the number of species defined for the fluid_system.
                The indices of the last dimension correspond to the indices in
                the active_species list of the fluid_system.
            fs: Reference to a fluid_system. Is optional if composition is
                defined by a dict. If not specified a new fluid_system with
                the components from the dict is created.
            shape: Tuple or list for the dimensions the fluid array. Can
                only be used if composition argument is a dict. Otherwise
                the dimensions are specified by the composition argument.
        """
        if fs is None:
            assert isinstance(composition, dict), 'fluid system must be specified if composition is not a dict'
            fs = fluid_system(list(composition.keys()))

        if isinstance(composition, list):
            composition = np.array(composition)

        if isinstance(composition, dict):
            missing_species = [s for s in composition if s not in fs.species]
            if len(missing_species):
                raise Exception(f'Species {missing_species[0]} is not part of the fluid system')

            species_composition = [composition[s] if s in composition.keys() else 0 for s in fs.species]

            comp_array = np.array(species_composition, dtype=NDFloat)
            if shape is not None:
                comp_array = comp_array * np.ones(list(shape) + [len(fs.species)], dtype=NDFloat)

        else:
            assert shape is None, 'specify shape by the shape of the composition array.'
            assert composition.shape[-1] == len(fs.species), f'composition.shape[-1] ({composition.shape[-1]}) must be {len(fs.species)}'
            comp_array = composition

        self.array_composition: FloatArray = comp_array
        self.total: FloatArray | float = np.sum(self.array_composition, axis=-1, dtype=NDFloat)
        self.array_fractions: FloatArray = self.array_composition / (np.expand_dims(self.total, -1) + _epsy)
        self.shape: _Shape = self.array_composition.shape[:-1]
        self.fs = fs
        self.array_elemental_composition: FloatArray = np.dot(self.array_composition, fs.array_species_elements)
        self.species = fs.species
        self.elements = fs.elements

    def get_composition_dict(self) -> dict[str, float]:
        """Get a dict of the molar amount of each fluid species

        Returns:
            Returns a dict of floats with the molar amount of each fluid species in mol
        """
        return {s: c for s, c in zip(self.fs.species, self.array_composition)}

    def get_fractions_dict(self) -> dict[str, float]:
        """Get a dict of the molar fractions of each fluid species

        Returns:
            Returns a dict of floats with the molar fractions of each fluid species
        """
        return {s: c for s, c in zip(self.fs.species, self.array_fractions)}

    def get_h(self, t: float | FloatArray) -> FloatArray | float:
        """Get specific enthalpy of the fluid at the given temperature

        Enthalpy is referenced to 25 °C and includes enthalpy of formation.
        Therefore the enthalpy of H2 and O2 is 0 at 25 °C, but the enthalpy
        of water vapor at 25 °C is −241 kJ/mol (enthalpy of formation).

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Enthalpies in J/mol
        """
        return np.sum(self.fs.get_species_h(t) * self.array_fractions, axis=-1, dtype=NDFloat)

    def get_H(self, t: float | FloatArray) -> FloatArray | float:
        """Get absolute enthalpy of the fluid at the given temperature

        Enthalpy is referenced to 25 °C and includes enthalpy of formation.
        Therefore the enthalpy of H2 and O2 is 0 at 25 °C, but the enthalpy
        of water vapor at 25 °C is −241 kJ/mol (enthalpy of formation).

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Enthalpies in J
        """
        return np.sum(self.fs.get_species_h(t) * self.array_composition, axis=-1, dtype=NDFloat)

    def get_s(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get molar entropy of the fluid at the given temperature and pressure

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Entropy in J/mol/K
        """
        x = self.array_fractions
        s = self.fs.get_species_s(t)

        return np.sum(x * (s - R * np.log(np.expand_dims(p / p0, -1) * x + _epsy)), axis=-1, dtype=NDFloat)

    def get_S(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get absolute entropy of the fluid at the given temperature and pressure

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Entropy in J/K
        """
        x = self.array_fractions
        n = self.array_composition
        s = self.fs.get_species_s(t)

        return np.sum(n * (s - R * np.log(np.expand_dims(p / p0, -1) * x + _epsy)), axis=-1, dtype=NDFloat)

    def get_cp(self, t: float | FloatArray) -> FloatArray | float:
        """Get molar heat capacity at constant pressure

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Heat capacity in J/mol/K
        """
        return np.sum(self.fs.get_species_cp(t) * self.array_fractions, axis=-1, dtype=NDFloat)

    def get_g(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get molar gibbs free energy (h - Ts)

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable
            p: Pressures(s) in Pascal. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Gibbs free energy in J/mol
        """
        x = self.array_fractions
        grt = self.fs.get_species_g_rt(t)

        return R * t * np.sum(x * (grt + np.log(np.expand_dims(p / p0, -1) * x + _epsy)), axis=-1, dtype=NDFloat)

    def get_G(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get absolute gibbs free energy (H - TS)

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable
            p: Pressures(s) in Pascal. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Gibbs free energy in J
        """
        x = self.array_fractions
        n = self.array_composition
        grt = self.fs.get_species_g_rt(t)

        return R * t * np.sum(n * (grt + np.log(np.expand_dims(p / p0, -1) * x + _epsy)), axis=-1, dtype=NDFloat)

    def get_g_rt(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get specific gibbs free energy divided by RT: g/R/T == (h/T-s)/R

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable
            p: Pressures(s) in Pascal. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Gibbs free energy divided by RT (dimensionless)
        """
        x = self.array_fractions
        grt = self.fs.get_species_g_rt(t)

        return np.sum(x * (grt + np.log(np.expand_dims(p / p0, -1) * x + _epsy)), axis=-1, dtype=NDFloat)

    def get_v(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get Absolute fluid volume

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable
            p: Pressure in Pa. Fluid shape and shape of the pressure
                must be broadcastable

        Returns:
            Volume of the fluid in m³
        """
        return R / p * t * self.total

    def get_vm(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get molar fluid volume

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable
            p: Pressure in Pa. Fluid shape and shape of the pressure
                must be broadcastable

        Returns:
            Molar volume of the fluid in m³/mol
        """
        return R / p * t

    def get_mass(self) -> FloatArray | float:
        """Get Absolute fluid mass

        Returns:
            Mass of the fluid in kg
        """
        return np.sum(self.array_composition * self.fs.array_molar_mass, axis=-1, dtype=NDFloat)

    def get_molar_mass(self) -> FloatArray | float:
        """Get molar fluid mass

        Returns:
            Mass of the fluid in kg/mol
        """
        return np.sum(self.array_fractions * self.fs.array_molar_mass, axis=-1, dtype=NDFloat)

    def get_density(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get mass based fluid density

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable
            p: Pressure in Pa. Fluid shape and shape of the pressure
                must be broadcastable

        Returns:
            Density of the fluid in kg/m³
        """
        return np.sum(self.array_fractions * self.fs.array_molar_mass, axis=-1, dtype=NDFloat) / (R * t) * p

    def get_x(self, species: str | list[str] | None = None) -> FloatArray:
        """Get molar fractions of fluid species

        Args:
            species: A single species name, a list of species names or None for
                returning the molar fractions of all species

        Returns:
            Returns an array of floats with the molar fractions of the species.
            If the a single species name is provided the return float array has
            the same dimensions as the fluid type. If a list or None is provided
            the return array has an additional dimension for the species.
        """
        if not species:
            return self.array_fractions
        elif isinstance(species, str):
            assert species in self.fs.species, f'Species {species} is not part of the fluid system'
            return self.array_fractions[..., self.fs.species.index(species)]
        else:
            assert set(species) <= set(self.fs.species), f'Species {", ".join([s for s in species if s not in self.fs.species])} is/are not part of the fluid system'
            return self.array_fractions[..., [self.fs.species.index(k) for k in species]]

    def __add__(self, other: _T) -> _T:
        return _array_operation(self, other, np.add)

    def __sub__(self, other: _T) -> _T:
        return _array_operation(self, other, np.subtract)

    def __truediv__(self, other: int | float | NDArray[Any]) -> 'fluid':
        if isinstance(other, np.ndarray):
            k = np.expand_dims(other, -1)
        else:
            k = np.array(other, dtype=NDFloat)
        return self.__class__(self.array_composition / k, self.fs)

    def __mul__(self, other: int | float | NDArray[Any]) -> 'fluid':
        k = np.expand_dims(other, -1) if isinstance(other, np.ndarray) else other
        return self.__class__(self.array_composition * k, self.fs)

    def __rmul__(self, other: int | float | NDArray[Any]) -> 'fluid':
        k = np.expand_dims(other, -1) if isinstance(other, np.ndarray) else other
        return self.__class__(self.array_composition * k, self.fs)

    def __neg__(self) -> 'fluid':
        return self.__class__(-self.array_composition, self.fs)

    # def __array__(self) -> FloatArray:
    #     return self.array_composition

    def __getitem__(self, key: str | int | list[str] | list[int] | slice) -> FloatArray:
        if isinstance(key, str):
            assert key in self.fs.species, f'Species {key} is not part of the fluid system'
            return self.array_composition[..., self.fs.species.index(key)]
        elif isinstance(key, (slice, int)):
            return self.array_composition[..., key]
        else:
            mset = set(self.fs.species) | set(range(len(self.fs.species)))
            assert set(key) <= mset, f'Species {", ".join([str(s) for s in key if s not in mset])} is/are not part of the fluid system'
            return self.array_composition[..., [self.fs.species.index(k) if isinstance(k, str) else k for k in key]]

    def __iter__(self) -> Iterator[dict[str, float]]:
        return iter({s: c for s, c in zip(self.fs.species, spa)} for spa in np.array(self.array_composition, ndmin=2))

    def __repr__(self) -> str:
        if len(self.array_fractions.shape) == 1:
            lines = [f'{s:16} {c * 100:5.2f} %' for s, c in zip(self.fs.species, self.array_fractions)]
            return f'{"Total":16} {self.total:8.3e} mol\n' + '\n'.join(lines)
        else:
            array_disp = self.array_fractions.__repr__()
            padding = int(array_disp.find('\n') / (len(self.fs.species) + 1))
            return ('Total mol:\n' + self.total.__repr__() +
                    '\nSpecies:\n' + ' ' * int(padding / 2) +
                    ''.join([(' ' * (padding - len(s))) + s for s in self.fs.species]) +
                    '\nMolar fractions:\n' + self.array_fractions.__repr__())


class elements:
    """Represent a fluid by composition of elements.

    Attributes:
        array_element_composition: Array of the element composition
    """

    __array_priority__ = 100

    def __init__(self, composition: fluid | dict[str, float] | list[str] | list[float] | FloatArray,
                 fs: fluid_system | None = None, shape: Sequence[int] | None = None):
        """Instantiates an elements object.

        Args:
            composition: A fluid object, a dict of element names with their
                composition, e.g.: {'O':1,'H':2} or a list/numpy-array of compositions.
                The array can be multidimensional, the size of the last dimension
                must match the number of elements used in the fluid_system.
                The indices of the last dimension correspond to the indices in
                the active_species list of the fluid_system.
            fs: Reference to a fluid_system.
            shape: Tuple or list for the dimensions the fluid array. Can
                only be used if composition argument is a dict. Otherwise
                the dimensions are specified by the composition argument.
        """
        if isinstance(composition, list):
            composition = np.array(composition)

        if isinstance(composition, fluid):
            new_composition: FloatArray = np.dot(composition.array_composition, composition.fs.array_species_elements)
            if fs:
                self.array_elemental_composition = _reorder_array(new_composition, composition.fs.elements, fs.elements)
            else:
                self.array_elemental_composition = new_composition
                fs = composition.fs
        elif isinstance(composition, dict) and fs is None:
            fs = fluid_system(species(element_names=list(composition.keys())))
        else:
            assert fs, 'fluid system must be specified if composition is not specified by a fluid'

        if isinstance(composition, dict):
            missing_elements = [s for s in composition if s not in fs.elements]
            if len(missing_elements):
                raise Exception(f'Element {missing_elements[0]} is not part of the fluid system')

            self.array_elemental_composition = np.array([composition[s] if s in composition.keys() else 0 for s in fs.elements])

            if shape is not None:
                self.array_elemental_composition = self.array_elemental_composition * np.ones(list(shape) + [len(fs.species)])

        elif isinstance(composition, np.ndarray):
            assert shape is None, 'specify shape by the shape of the composition array.'
            assert composition.shape[-1] == len(fs.elements), f'composition.shape[-1] ({composition.shape[-1]}) must be {len(fs.elements)}'
            self.array_elemental_composition = composition

        self.shape: _Shape = self.array_elemental_composition.shape[:-1]
        self.fs = fs
        self.elements = fs.elements

    def get_elemental_composition(self) -> dict[str, float]:
        """Get a dict of the molar amount of each element

        Returns:
            Returns a dict of floats with the molar amount of each element in mol
        """
        return {s: c for s, c in zip(self.fs.elements, self.array_elemental_composition)}

    def get_mass(self) -> FloatArray | float:
        """Get absolute mass of elements

        Returns:
            Mass of the fluid in kg
        """
        return np.sum(self.array_elemental_composition * self.fs.array_atomic_mass, axis=-1, dtype=NDFloat)

    def __add__(self, other: 'fluid | elements') -> 'elements':
        return _array_operation(self, other, np.add)

    def __sub__(self, other: 'fluid | elements') -> 'elements':
        return _array_operation(self, other, np.subtract)

    def __truediv__(self, other: int | float | FloatArray) -> 'elements':
        k = np.expand_dims(other, -1) if isinstance(other, np.ndarray) else other
        ttes = self.array_elemental_composition / k
        return self.__class__(self.array_elemental_composition / k + ttes, self.fs)

    def __mul__(self, other: int | float | FloatArray) -> 'elements':
        k = np.expand_dims(other, -1) if isinstance(other, np.ndarray) else other
        return self.__class__(self.array_elemental_composition * k, self.fs)

    def __rmul__(self, other: int | float | FloatArray) -> 'elements':
        k = np.expand_dims(other, -1) if isinstance(other, np.ndarray) else other
        return self.__class__(self.array_elemental_composition * k, self.fs)

    def __neg__(self) -> 'elements':
        return self.__class__(-self.array_elemental_composition, self.fs)

    def __array__(self) -> FloatArray:
        return self.array_elemental_composition

    def __getitem__(self, key: str | int | list[str] | list[int] | slice) -> FloatArray:
        if isinstance(key, str):
            assert key in self.fs.elements, f'Element {key} is not part of the fluid system'
            return self.array_elemental_composition[..., self.fs.elements.index(key)]
        elif isinstance(key, (slice, int)):
            return self.array_elemental_composition[..., key]
        else:
            mset = set(self.fs.elements) | set(range(len(self.fs.elements)))
            assert set(key) <= mset, f'Elements {", ".join([str(s) for s in key if s not in mset])} is/are not part of the fluid system'
            return self.array_elemental_composition[..., [self.fs.elements.index(k) if isinstance(k, str) else k for k in key]]

    def __iter__(self) -> Iterator[dict[str, float]]:
        return iter({s: c for s, c in zip(self.fs.elements, spa)} for spa in np.array(self.array_elemental_composition, ndmin=2))

    def __repr__(self) -> str:
        if len(self.array_elemental_composition.shape) == 1:
            lines = [f'{s:16} {c:5.3e} mol' for s, c in zip(self.fs.elements, self.array_elemental_composition)]
            return '\n'.join(lines)
        else:
            array_disp = self.array_elemental_composition.__repr__()
            padding = int(array_disp.find('\n') / (len(self.fs.elements) + 1))
            return ('Elements:\n' + ' ' * int(padding / 2) +
                    ''.join([(' ' * (padding - len(s))) + s for s in self.fs.elements]) +
                    '\nMols:\n' + self.array_elemental_composition.__repr__())


# def _combine_index(index1: list[str], index2: list[str]) -> list[str]:
#     return list(set(index1) | set(index2))


def _reorder_array(arr: FloatArray, old_index: list[str], new_index: list[str]) -> FloatArray:
    """Reorder the last dimension of an array according to a provided list of species
    names in the old oder and a list in the new order.

    Args:
        arr: Array to be reordered
        old_index: List of species names in the current order
        new_index: List of species names in the new order

    Returns:
        Array with the last dimension reordered
    """
    ret_array = np.zeros([*arr.shape[:-1], len(new_index)])
    for i, k in enumerate(old_index):
        ret_array[..., new_index.index(k)] = arr[..., i]
    return ret_array


@overload
def _array_operation(self: elements, other: elements | fluid, func: Callable[[FloatArray, FloatArray], FloatArray]) -> elements:
    pass


@overload
def _array_operation(self: fluid, other: _T, func: Callable[[FloatArray, FloatArray], FloatArray]) -> _T:
    pass


@overload
def _array_operation(self: _T, other: fluid, func: Callable[[FloatArray, FloatArray], FloatArray]) -> _T:
    pass


def _array_operation(self: elements | fluid, other: elements | fluid, func: Callable[[FloatArray, FloatArray], FloatArray]) -> elements | fluid:
    """Perform an array operation on two fluid or elements objects.
    The operation is provided by a Callable that takes two arguments.

    Args:
        self: First fluid or elements object
        other: Second fluid or elements object
        func: Callable function to perform the operation

    Returns:
        A new fluid or elements object with the result of the
    """
    assert isinstance(other, elements) or isinstance(other, fluid)
    if self.fs is other.fs:
        if isinstance(self, elements) or isinstance(other, elements):
            return elements(func(self.array_elemental_composition, other.array_elemental_composition), self.fs)
        else:
            return fluid(func(self.array_composition, other.array_composition), self.fs)
    elif set(self.fs.species) >= set(other.fs.species):
        if isinstance(self, elements) or isinstance(other, elements):
            el_array = _reorder_array(other.array_elemental_composition, other.fs.elements, self.fs.elements)
            return elements(func(self.array_elemental_composition, el_array), self.fs)
        else:
            el_array = _reorder_array(other.array_composition, other.fs.species, self.fs.species)
            return fluid(func(self.array_composition, el_array), self.fs)
    elif set(self.fs.species) < set(other.fs.species):
        if isinstance(self, elements) or isinstance(other, elements):
            el_array = _reorder_array(self.array_elemental_composition, self.fs.elements, other.fs.elements)
            return elements(func(el_array, other.array_elemental_composition), other.fs)
        else:
            el_array = _reorder_array(self.array_composition, self.fs.species, other.fs.species)
            return fluid(func(el_array, other.array_composition), other.fs)
    else:
        new_fs = fluid_system(sorted(list(set(self.fs.species) | set(other.fs.species))))
        if isinstance(self, elements) or isinstance(other, elements):
            el_array1 = _reorder_array(self.array_elemental_composition, self.fs.elements, new_fs.elements)
            el_array2 = _reorder_array(other.array_elemental_composition, other.fs.elements, new_fs.elements)
            return elements(func(el_array1, el_array2), new_fs)
        else:
            el_array1 = _reorder_array(self.array_composition, self.fs.species, new_fs.species)
            el_array2 = _reorder_array(other.array_composition, other.fs.species, new_fs.species)
            return fluid(func(el_array1, el_array2), new_fs)


def stack(arrays: list[_T], axis: int = 0) -> _T:
    """Stack a list of fluid or elements objects along a new axis

    Args:
        arrays: List of arrays
        axis: Axis to stack the fluid objects along

    Returns:
        A new array object stacked along the new axis
    """
    a0 = arrays[0]
    assert all(a.fs == a0.fs for a in arrays), 'All objects must have the same fluid system'
    assert axis <= len(a0.shape), f'Axis must be smaller or equal to len(shape) ({len(a0.shape)})'
    return a0.__class__(np.stack(
        [a.array_elemental_composition if isinstance(a, elements) else a.array_composition for a in arrays],
        axis=axis), a0.fs)


def concat(arrays: list[_T], axis: int = 0) -> _T:
    """Concatenate a list of fluid or elements objects along an existing axis

    Args:
        arrays: List of arrays
        axis: Axis to concatenate the fluid objects along

    Returns:
        A new array object stacked along the specified axis
    """
    a0 = arrays[0]
    assert all(f.fs == a0.fs for f in arrays), 'All fluid objects must have the same fluid system'
    assert axis < len(a0.shape), f'Axis must be smaller than shape len({a0.shape})'
    return a0.__class__(np.concatenate(
        [a.array_elemental_composition if isinstance(a, elements) else a.array_composition for a in arrays],
        axis=axis), a0.fs)


def _equilibrium_gmin(fs: fluid_system, element_composition: FloatArray, t: float, p: float) -> FloatArray:
    """Calculate the equilibrium composition of a fluid based on minimizing the Gibbs free energy"""
    def element_balance(n: FloatArray, fs: fluid_system, ref: FloatArray) -> FloatArray:
        return np.dot(n, fs.array_species_elements) - ref  # type: ignore

    def gibbs_rt(n: FloatArray, grt: FloatArray, p_rel: float):  # type: ignore
        # Calculate G/(R*T)
        return np.sum(n * (grt + np.log(p_rel * n / np.sum(n) + _epsy)))

    cons: dict[str, Any] = {'type': 'eq', 'fun': element_balance, 'args': [fs, element_composition]}
    bnds = [(0, None) for _ in fs.species]
    grt = fs.get_species_g_rt(t)
    p_rel = p / p0

    start_composition_array = np.ones_like(fs.species, dtype=float)
    sol = np.array(minimize(gibbs_rt, start_composition_array, args=(grt, p_rel), method='SLSQP',
                   bounds=bnds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})['x'], dtype=NDFloat)  # type: ignore

    return sol


def _equilibrium_eq(fs: fluid_system, element_composition: FloatArray, t: float, p: float) -> FloatArray:
    """Calculate the equilibrium composition of a fluid based on equilibrium equations"""
    el_max = np.max(element_composition)
    element_norm = element_composition / el_max

    a = fs.array_stoichiometric_coefficients
    a_sum = np.sum(a)
    el_matrix = fs.array_species_elements.T

    # Log equilibrium constants for each reaction equation
    b = -np.sum(fs.get_species_g_rt(t) * a, axis=1)

    # Pressure corrected log equilibrium constants
    bp = b - np.sum(a * np.log(p / p0), axis=1)

    logn_start = np.ones(el_matrix.shape[1]) * 0.1

    def residuals(logn: FloatArray):  # type: ignore
        n = np.exp(logn)
        n_sum = np.sum(n)

        # Residuals from equilibrium equations:
        eq_resid = np.dot(a, logn - np.log(n_sum)) - bp

        # Derivative:
        j_eq = a - a_sum * n / n_sum

        # Residuals from elemental balance:
        el_error = np.dot(el_matrix, n) - element_norm
        ab_resid = np.log1p(el_error)

        # Derivative:
        j_ab = el_matrix * n / np.expand_dims(el_error + 1, axis=1)

        return (np.hstack([eq_resid, ab_resid]), np.concatenate([j_eq, j_ab], axis=0))

    ret = root(residuals, logn_start, jac=True, tol=1e-30)
    n = np.exp(np.array(ret['x'], dtype=NDFloat))

    return n * el_max


def equilibrium(f: fluid | elements, t: float | FloatArray, p: float = 1e5) -> fluid:
    """Calculate the equilibrium composition of a fluid at a given temperature and pressure"

    Args:
        f: Fluid or elements object
        t: Temperature in Kelvin
        p: Pressure in Pascal

    Returns:
        A new fluid object with the equilibrium composition
    """
    assert isinstance(f, (fluid, elements)), 'Argument f must be a fluid or elements'
    m_shape: int = f.fs.array_stoichiometric_coefficients.shape[0]
    if isinstance(f, fluid):
        if not m_shape:
            return f
    else:
        if not m_shape:
            def linalg_lstsq(array_elemental_composition: FloatArray, matrix: FloatArray) -> Any:
                # TODO: np.dot(np.linalg.pinv(a), b) is eqivalent to lstsq(a, b).
                # the constant np.linalg.pinv(a) can be precomputed for each fs.
                return np.dot(np.linalg.pinv(matrix), array_elemental_composition)

            print('-->', f.array_elemental_composition.shape, f.fs.array_species_elements.transpose().shape)
            composition = np.apply_along_axis(linalg_lstsq, -1, f.array_elemental_composition, f.fs.array_species_elements.transpose())
            return fluid(composition, f.fs)

    assert np.min(f.array_elemental_composition) >= 0, 'Input element fractions must be 0 or positive'
    if isinstance(t, np.ndarray):
        assert f.shape == tuple(), 'Multidimensional temperature can currently only used for 0D fluids'
        t_composition = np.zeros(t.shape + (f.fs.array_species_elements.shape[0],))
        for t_index in np.ndindex(t.shape):
            t_composition[t_index] = _equilibrium_solver(f.fs, f.array_elemental_composition, t[t_index], p)
        return fluid(t_composition, f.fs)
    else:
        composition = np.ones(f.shape + (len(f.fs.species),), dtype=float)
        for index in np.ndindex(f.shape):
            #print(composition.shape, index, _equilibrium(f.fs, f._element_composition[index], t, p))
            composition[index] = _equilibrium_solver(f.fs, f.array_elemental_composition[index], t, p)
        return fluid(composition, f.fs)


def carbon_activity(f: fluid | elements, t: float, p: float) -> float:
    """Calculate the activity of carbon in a fluid at a given temperature and pressure.
    At a value of 1 the fluid is in equilibrium with solid graphite. At a value > 1
    additional carbon formation is thermodynamic favored. At a value < 1 a
    depletion of solid carbon is favored.

    Args:
        f: Fluid or elements object
        t: Temperature in Kelvin
        p: Pressure in Pascal

    Returns:
        The activity of carbon in the fluid
    """
    # Values for solid state carbon (graphite) from NIST-JANAF Tables
    # https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Carbon.pdf
    # https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-1Vol1-Intro.pdf
    # Polynomial is valid for T from 100 to 2500 K
    pgef = np.array([-6.76113852E-02, 2.02187857E+00, -2.38664605E+01,
                    1.43575462E+02, -4.51375503E+02, 6.06219665E+02])

    # Gibbs free energy divided by RT for carbon
    g_rtc = -np.sum(pgef * np.log(np.expand_dims(t, -1))**np.array([5, 4, 3, 2, 1, 0])) / R

    g_rt = f.fs.get_species_g_rt(t)

    x = equilibrium(f, t, p).array_fractions

    i_co = f.fs.species.index('CO')
    i_co2 = f.fs.species.index('CO2')
    i_h2 = f.fs.species.index('H2')
    i_h2o = f.fs.species.index('H2O')
    i_ch4 = f.fs.species.index('CH4')

    if min(x[i_co], x[i_co2]) > min([x[i_ch4], x[i_h2o], x[i_h2]]) and min(x[i_co], x[i_co2]) > 0:
        # 2 CO -> CO2 + C(s) (Boudouard reaction)
        lnalpha = (2 * g_rt[i_co] - (g_rt[i_co2] + g_rtc)) + np.log(
            x[i_co]**2 / x[i_co2] * (p / p0))
    elif min([x[i_ch4], x[i_h2o], x[i_co]]) > 1E-4:
        # CH4 + 2 CO -> 2 H2O + 3 C(s)
        lnalpha = ((g_rt[i_ch4] + 2 * g_rt[i_co] - 3 * g_rtc - 2 * g_rt[i_h2o]) + np.log(
            x[i_ch4] * x[i_co]**2 / x[i_h2o]**2 * (p / p0))) / 3
    elif min(x[i_h2], x[i_ch4]) > 0:
        # if x[i_h2] or x[i_ch4] is small compared to the precision of the
        # component concentrations the result can be inaccurate
        # CH4 -> 2 H2 + C(s)
        # CH4 + CO2 -> 2 H2 + 2 CO
        # 2 H2O - O2 -> 2 H2
        lnalpha = (g_rt[i_ch4] - (2 * g_rt[i_h2] + g_rtc)) + np.log(
            x[i_ch4] / x[i_h2]**2 / (p / p0))
    elif x[i_h2] == 0:
        # equilibrium on carbon side
        lnalpha = 10
    else:
        # equilibrium on non-carbon side
        lnalpha = -10

    return exp(lnalpha)


def oxygen_partial_pressure(f: fluid | elements, t: float, p: float) -> FloatArray | float:
    _oxygen_data = fluid({'O2': 1})

    def get_oxygen(x: FloatArray) -> float:
        g_rt = f.fs.get_species_g_rt(t)
        g_rt_o2 = _oxygen_data.fs.get_species_g_rt(t)[0]

        i_co = f.fs.species.index('CO') if 'C' in f.fs.elements else None
        i_co2 = f.fs.species.index('CO2') if 'C' in f.fs.elements else None
        i_o2 = f.fs.species.index('O2') if 'O2' in f.fs.species else None
        i_h2o = f.fs.species.index('H2O') if 'H' in f.fs.elements else None
        i_h2 = f.fs.species.index('H2') if 'H' in f.fs.elements else None
        i_ch4 = f.fs.species.index('CH4') if 'CH4' in f.fs.species else None

        ox_ref_val = max([float(v) for v in (x[i_h2] * x[i_h2o], x[i_co2] * x[i_co], x[i_ch4]) if v.shape == tuple()] + [0])

        # print([i_o2, x[i_o2], ox_ref_val])

        if i_o2 is not None and x[i_o2] > ox_ref_val:
            # print('o O2')
            return float(x[i_o2] * p)

        elif i_ch4 is not None and x[i_ch4] > x[i_co2] * 100 and x[i_ch4] > x[i_h2o] * 100:
            # print('o ch4')
            # 2CH4 + O2 <--> 4H2 + 2CO
            lnpo2 = 4 * g_rt[i_h2] + 2 * g_rt[i_co] - 2 * g_rt[i_ch4] - g_rt_o2 + np.log(x[i_h2]**4 * x[i_co]**2 / x[i_ch4]**2) - 2 * np.log(p / p0)

        elif (i_co is None and i_h2 is not None) or (i_h2 is not None and i_co is not None and (x[i_h2] * x[i_h2o] > x[i_co2] * x[i_co])):
            # print('o h2o')
            # 2H2 + O2 <--> 2H2O
            lnpo2 = 2 * (g_rt[i_h2o] - g_rt[i_h2] + np.log(x[i_h2o] / x[i_h2])) - g_rt_o2 - np.log(p / p0)

        else:
            assert i_co is not None
            # print('o co2')
            # 2CO + O2 <--> 2CO2
            lnpo2 = 2 * (g_rt[i_co2] - g_rt[i_co] + np.log(x[i_co2] / x[i_co])) - g_rt_o2 - np.log(p / p0)

        return exp(lnpo2) * p

    x = equilibrium(f, t, p).array_fractions

    if len(x.shape):
        return np.apply_along_axis(get_oxygen, -1, x)
    else:
        return get_oxygen(x)


_equilibrium_solver = _equilibrium_eq
