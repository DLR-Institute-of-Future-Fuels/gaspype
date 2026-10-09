import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Any, TypeVar, Iterator, overload, Callable
from types import ModuleType
from math import log as ln, ceil
import operator
from ._numerics import null_space
from ._backend import get_backend, as_float_array
from gaspype._phys_data import atomic_weights, db_reader
import re
import pkgutil
from .constants import R, epsy, p0
from .typing import FloatArray, NDFloat, Shape, ArrayIndices

T = TypeVar('T', 'fluid', 'elements')

_data = pkgutil.get_data(__name__, 'data/therm_data.bin')
assert _data is not None, 'Could not load thermodynamic data'
_species_db = db_reader(_data)


def species(pattern: str = '*', element_names: str | list[str] = [], use_regex: bool = False) -> list[str]:
    """Returns a alphabetically sorted list of all available species
    filtered by a pattern if supplied

    Args:
        pattern: Optional filter for specific molecules. Placeholder
            characters: # A number including non written ones: 'C#H#' matches 'CH4';
            $ Arbitrary element name; * Any sequence of characters
        element_names: restrict results to species that contain only the specified elements.
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


class fluid_system:
    """A class to represent a fluid_system defined by a set of selected species.

    Attributes:
        species_names (list[str]): List of selected species in the fluid_system
        array_molar_mass (FloatArray): Array of the molar masses of the species in the fluid_system
        array_element_composition (FloatArray): Array of the element composition of the species in the fluid_system.
            Dimension is: (number of species, number of elements)
        array_atomic_mass (FloatArray): Array of the atomic masses of the elements in the fluid_system
        np (ModuleType): Array namespace of the backend, used for all arrays and calculations
            of the fluid_system and of the fluids and elements based on it
        device (Any): JAX device of the arrays or None for the default device
    """

    def __init__(self, species: list[str] | str, t_min: int = 250, t_max: int = 2000,
                 backend: str | ModuleType = 'numpy', device: Any = None):
        """Instantiates a fluid_system.

        Args:
            species: List of species names to be available in the constructed
                fluid_system (as list of strings or a comma separated string)
            t_min: Lower bound of the required temperature range in Kelvin
            t_max: Upper bound of the required temperature range in Kelvin
            backend: Array library used for the fluid_system and for all fluids
                and elements based on it: 'numpy', 'jax' or the module
                of a NumPy compatible array library (e.g. jax.numpy)
            device: Optional device for the JAX backend: A JAX device or
                a string like 'cpu', 'gpu' or 'gpu:1'
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
        self.np, self.device = get_backend(backend, device)
        self.species = species
        self.active_species = species  # for backward compatibility
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
        self.array_species_elements: FloatArray = np.array([[ec[el] if el in ec else 0.0 for el in self.elements] for ec in element_compositions])

        self.array_atomic_mass: FloatArray = np.array([atomic_weights[el] for el in self.elements]) * 1e-3  # kg/mol
        self.array_molar_mass: FloatArray = np.sum(self.array_atomic_mass * self.array_species_elements, axis=-1)  # kg/mol

        self.array_stoichiometric_coefficients: FloatArray = np.array(null_space(self.array_species_elements.T), dtype=NDFloat).T

        if self.np is not np:
            for name in ['_cp_array', '_h_array', '_s_array', '_g_rt_array', 'array_species_elements',
                         'array_atomic_mass', 'array_molar_mass', 'array_stoichiometric_coefficients']:
                setattr(self, name, self._asarray(getattr(self, name)))

    def _asarray(self, a: Any) -> Any:
        """Convert a scalar, list or array to a floating point array of
        the backend and on the device of the fluid system"""
        return as_float_array(a, self.np, self.device)

    def get_species_h(self, t: float | FloatArray) -> FloatArray:
        """Get the molar enthalpies for all species in the fluid system

        Args:
            t: Temperature in Kelvin (can be an array)

        Returns:
            Array with the enthalpies of each specie in J/mol
        """
        return lookup(self._h_array, self._asarray(t), self._t_offset, self.np)

    def get_species_s(self, t: float | FloatArray) -> FloatArray:
        """Get the molar entropies for all species in the fluid system

        Args:
            t: Temperature in Kelvin (can be an array)

        Returns:
            Array with the entropies of each specie in J/mol/K
        """
        return lookup(self._s_array, self._asarray(t), self._t_offset, self.np)

    def get_species_cp(self, t: float | FloatArray) -> FloatArray:
        """Get the isobaric molar heat capacity for all species in the fluid system

        Args:
            t: Temperature in Kelvin (can be an array)

        Returns:
            Array with the heat capacities of each specie in J/mol/K
        """
        return lookup(self._cp_array, self._asarray(t), self._t_offset, self.np)

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
        return lookup(self._g_rt_array, self._asarray(t), self._t_offset, self.np)

    def get_species_references(self) -> str:
        """Get a string with the references for all fluids of the fluid system

        Returns:
            String with the references
        """
        return '\n'.join([f'{s:<12}: {_species_db[s].ref_string}' for s in self.species])

    def __add__(self, other: 'fluid_system') -> 'fluid_system':
        assert isinstance(other, self.__class__)
        return self.__class__(self.species + other.species, backend=self.np, device=self.device)

    def __repr__(self) -> str:
        return ('Fluid system\n    Species:  ' + ', '.join(self.species) +
                '\n    Elements: ' + ', '.join(self.elements))


class fluid:
    """A class to represent a fluid defined by a composition of
    one or more species.

    Attributes:
        species (list[str]): List of species names in the associated fluid_system
        array_composition (FloatArray): Array of the molar amounts of the species in the fluid
        array_element_composition (FloatArray): Array of the element composition in the fluid
        array_fractions (FloatArray): Array of the molar fractions of the species in the fluid
        total (FloatArray | float): Array of the sums of the molar amount of all species
        fs (fluid_system): Reference to the fluid_system used for this fluid
        shape (_Shape): Shape of the fluid array
        elements (list[str]): List of elements in the fluid_system
    """

    __array_priority__ = 100

    def __init__(self, composition: dict[str, float] | list[float] | FloatArray,
                 fs: fluid_system | None = None,
                 shape: Sequence[int] | None = None):
        """Instantiates a fluid.

        Args:
            composition: A dict of species names with their composition, e.g.
                {'O2':0.5,'H2O':0.5} or a list/numpy-array of compositions.
                The array can be multidimensional, the size of the last dimension
                must match the number of species defined for the fluid_system.
                The indices of the last dimension correspond to the indices in
                the species list of the fluid_system.
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

        np = fs.np

        if isinstance(composition, dict):
            missing_species = [s for s in composition if s not in fs.species]
            if len(missing_species):
                raise Exception(f'Species {missing_species[0]} is not part of the fluid system')

            species_composition = [composition[s] if s in composition.keys() else 0 for s in fs.species]

            comp_array = fs._asarray(species_composition)
            if shape is not None:
                comp_array = comp_array * np.ones((*shape, len(fs.species)))

        else:
            assert shape is None, 'specify shape by the shape of the composition array.'
            comp_array = fs._asarray(composition)
            assert comp_array.shape[-1] == len(fs.species), f'composition.shape[-1] ({comp_array.shape[-1]}) must be {len(fs.species)}'

        self.array_composition: FloatArray = comp_array
        self.total: FloatArray | float = np.sum(self.array_composition, axis=-1)
        self.array_fractions: FloatArray = self.array_composition / (np.expand_dims(self.total, -1) + epsy)
        self.shape: Shape = self.array_composition.shape[:-1]
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
        np = self.fs.np
        return np.sum(self.fs.get_species_h(t) * self.array_fractions, axis=-1)  # type: ignore[no-any-return]

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
        np = self.fs.np
        return np.sum(self.fs.get_species_h(t) * self.array_composition, axis=-1)  # type: ignore[no-any-return]

    def get_s(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get molar entropy of the fluid at the given temperature and pressure

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Entropy in J/mol/K
        """
        np = self.fs.np
        x = self.array_fractions
        s = self.fs.get_species_s(t)

        return np.sum(x * (s - R * np.log(np.expand_dims(p / p0, -1) * x + epsy)), axis=-1)  # type: ignore[no-any-return]

    def get_S(self, t: float | FloatArray, p: float | FloatArray) -> FloatArray | float:
        """Get absolute entropy of the fluid at the given temperature and pressure

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Entropy in J/K
        """
        np = self.fs.np
        x = self.array_fractions
        n = self.array_composition
        s = self.fs.get_species_s(t)

        return np.sum(n * (s - R * np.log(np.expand_dims(p / p0, -1) * x + epsy)), axis=-1)  # type: ignore[no-any-return]

    def get_cp(self, t: float | FloatArray) -> FloatArray | float:
        """Get molar heat capacity at constant pressure

        Args:
            t: Absolute temperature(s) in Kelvin. Fluid shape and shape of the temperature
                must be broadcastable

        Returns:
            Heat capacity in J/mol/K
        """
        np = self.fs.np
        return np.sum(self.fs.get_species_cp(t) * self.array_fractions, axis=-1)  # type: ignore[no-any-return]

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
        np = self.fs.np
        x = self.array_fractions
        grt = self.fs.get_species_g_rt(t)

        return R * t * np.sum(x * (grt + np.log(np.expand_dims(p / p0, -1) * x + epsy)), axis=-1)  # type: ignore[no-any-return]

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
        np = self.fs.np
        x = self.array_fractions
        n = self.array_composition
        grt = self.fs.get_species_g_rt(t)

        return R * t * np.sum(n * (grt + np.log(np.expand_dims(p / p0, -1) * x + epsy)), axis=-1)  # type: ignore[no-any-return]

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
        np = self.fs.np
        x = self.array_fractions
        grt = self.fs.get_species_g_rt(t)

        return np.sum(x * (grt + np.log(np.expand_dims(p / p0, -1) * x + epsy)), axis=-1)  # type: ignore[no-any-return]

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
        np = self.fs.np
        return np.multiply(R / p, t)  # type: ignore[no-any-return]

    def get_mass(self) -> FloatArray | float:
        """Get Absolute fluid mass

        Returns:
            Mass of the fluid in kg
        """
        np = self.fs.np
        return np.sum(self.array_composition * self.fs.array_molar_mass, axis=-1)  # type: ignore[no-any-return]

    def get_molar_mass(self) -> FloatArray | float:
        """Get molar fluid mass

        Returns:
            Mass of the fluid in kg/mol
        """
        np = self.fs.np
        return np.sum(self.array_fractions * self.fs.array_molar_mass, axis=-1)  # type: ignore[no-any-return]

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
        np = self.fs.np
        return np.sum(self.array_fractions * self.fs.array_molar_mass, axis=-1) / (R * t) * p  # type: ignore[no-any-return]

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

    def get_n(self, species: str | list[str] | None = None) -> FloatArray:
        """Get molar amount of fluid species

        Args:
            species: A single species name, a list of species names or None for
                returning the amount of all species

        Returns:
            Returns an array of floats with the molar amount of the species.
            If the a single species name is provided the return float array has
            the same dimensions as the fluid type. If a list or None is provided
            the return array has an additional dimension for the species.
        """
        if not species:
            return self.array_composition
        elif isinstance(species, str):
            assert species in self.fs.species, f'Species {species} is not part of the fluid system'
            return self.array_composition[..., self.fs.species.index(species)]
        else:
            assert set(species) <= set(self.fs.species), f'Species {", ".join([s for s in species if s not in self.fs.species])} is/are not part of the fluid system'
            return self.array_composition[..., [self.fs.species.index(k) for k in species]]

    def __add__(self, other: T) -> T:
        return array_operation(self, other, operator.add)

    def __sub__(self, other: T) -> T:
        return array_operation(self, other, operator.sub)

    def __truediv__(self, other: int | float | NDArray[Any]) -> 'fluid':
        np = self.fs.np
        k = other if isinstance(other, (int, float)) else np.expand_dims(other, -1)
        return self.__class__(self.array_composition / k, self.fs)

    def __mul__(self, other: int | float | NDArray[Any]) -> 'fluid':
        np = self.fs.np
        k = other if isinstance(other, (int, float)) else np.expand_dims(other, -1)
        return self.__class__(self.array_composition * k, self.fs)

    def __rmul__(self, other: int | float | NDArray[Any]) -> 'fluid':
        np = self.fs.np
        k = other if isinstance(other, (int, float)) else np.expand_dims(other, -1)
        return self.__class__(self.array_composition * k, self.fs)

    def __neg__(self) -> 'fluid':
        return self.__class__(-self.array_composition, self.fs)

    # def __array__(self) -> FloatArray:
    #     return self.array_composition

    @overload
    def __getitem__(self, key: str) -> FloatArray:
        pass

    @overload
    def __getitem__(self, key: ArrayIndices) -> 'fluid':
        pass

    def __getitem__(self, key: str | ArrayIndices) -> Any:
        if isinstance(key, str):
            assert key in self.fs.species, f'Species {key} is not part of the fluid system'
            return self.array_composition[..., self.fs.species.index(key)]
        else:
            key_tuple = key if isinstance(key, tuple) else (key,)
            return fluid(self.array_composition[key_tuple + (slice(None),)], self.fs)

    def __iter__(self) -> Iterator[dict[str, float]]:
        assert len(self.shape) < 2, 'Cannot iterate over species with more than one dimension'
        aec = self.array_composition.reshape(-1, len(self.fs.species))
        return iter({s: c for s, c in zip(self.fs.species, aec[i, :])} for i in range(aec.shape[0]))

    def __repr__(self) -> str:
        if len(self.array_fractions.shape) == 1:
            lines = [f'{s:16} {float(c) * 100:5.2f} %' for s, c in zip(self.fs.species, self.array_fractions)]
            return f'{"Total":16} {float(self.total):8.3e} mol\n' + '\n'.join(lines)
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
        array_element_composition (FloatArray): Array of the element composition
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
                the species list of the fluid_system.
            fs: Reference to a fluid_system.
            shape: Tuple or list for the dimensions the fluid array. Can
                only be used if composition argument is a dict. Otherwise
                the dimensions are specified by the composition argument.
        """
        if isinstance(composition, fluid):
            if fs:
                self.array_elemental_composition: FloatArray = fs._asarray(
                    reorder_array(composition.array_elemental_composition, composition.fs.elements, fs.elements, composition.fs.np))
            else:
                self.array_elemental_composition = composition.array_elemental_composition
                fs = composition.fs
        elif isinstance(composition, dict) and fs is None:
            fs = fluid_system(species(element_names=list(composition.keys())))
        else:
            assert fs, 'fluid system must be specified if composition is not specified by a fluid'

        if isinstance(composition, dict):
            missing_elements = [s for s in composition if s not in fs.elements]
            if len(missing_elements):
                raise Exception(f'Element {missing_elements[0]} is not part of the fluid system')

            self.array_elemental_composition = fs._asarray([composition[s] if s in composition.keys() else 0 for s in fs.elements])

            if shape is not None:
                self.array_elemental_composition = self.array_elemental_composition * fs.np.ones((*shape, len(fs.elements)))

        elif not isinstance(composition, fluid):
            assert shape is None, 'specify shape by the shape of the composition array.'
            self.array_elemental_composition = fs._asarray(composition)
            assert self.array_elemental_composition.shape[-1] == len(fs.elements), \
                f'composition.shape[-1] ({self.array_elemental_composition.shape[-1]}) must be {len(fs.elements)}'

        self.shape: Shape = self.array_elemental_composition.shape[:-1]
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
        np = self.fs.np
        return np.sum(self.array_elemental_composition * self.fs.array_atomic_mass, axis=-1)  # type: ignore[no-any-return]

    def get_n(self, elemental_species: str | list[str] | None = None) -> FloatArray:
        """Get molar amount of elements

        Args:
            elemental_species: A single element name, a list of element names or None for
                returning the amount of all element

        Returns:
            Returns an array of floats with the molar amount of the elements.
            If the a single element name is provided the return float array has
            the same dimensions as the fluid type. If a list or None is provided
            the return array has an additional dimension for the elements.
        """
        if not elemental_species:
            return self.array_elemental_composition
        elif isinstance(elemental_species, str):
            assert elemental_species in self.fs.elements, f'Element {elemental_species} is not part of the fluid system'
            return self.array_elemental_composition[..., self.fs.elements.index(elemental_species)]
        else:
            assert set(elemental_species) <= set(self.fs.elements), f'Elements {", ".join([s for s in elemental_species if s not in self.fs.elements])} is/are not part of the fluid system'
            return self.array_elemental_composition[..., [self.fs.elements.index(k) for k in elemental_species]]

    def __add__(self, other: 'fluid | elements') -> 'elements':
        return array_operation(self, other, operator.add)

    def __sub__(self, other: 'fluid | elements') -> 'elements':
        return array_operation(self, other, operator.sub)

    def __truediv__(self, other: int | float | FloatArray) -> 'elements':
        np = self.fs.np
        k = other if isinstance(other, (int, float)) else np.expand_dims(other, -1)
        return self.__class__(self.array_elemental_composition / k, self.fs)

    def __mul__(self, other: int | float | FloatArray) -> 'elements':
        np = self.fs.np
        k = other if isinstance(other, (int, float)) else np.expand_dims(other, -1)
        return self.__class__(self.array_elemental_composition * k, self.fs)

    def __rmul__(self, other: int | float | FloatArray) -> 'elements':
        np = self.fs.np
        k = other if isinstance(other, (int, float)) else np.expand_dims(other, -1)
        return self.__class__(self.array_elemental_composition * k, self.fs)

    def __neg__(self) -> 'elements':
        return self.__class__(-self.array_elemental_composition, self.fs)

    def __array__(self) -> FloatArray:
        return np.asarray(self.array_elemental_composition)

    @overload
    def __getitem__(self, key: str) -> FloatArray:
        pass

    @overload
    def __getitem__(self, key: ArrayIndices) -> 'elements':
        pass

    def __getitem__(self, key: str | ArrayIndices) -> Any:
        if isinstance(key, str):
            assert key in self.fs.elements, f'Element {key} is not part of the fluid system'
            return self.array_elemental_composition[..., self.fs.elements.index(key)]
        else:
            key_tuple = key if isinstance(key, tuple) else (key,)
            return elements(self.array_elemental_composition[key_tuple + (slice(None),)], self.fs)

    def __iter__(self) -> Iterator[dict[str, float]]:
        assert len(self.shape) < 2, 'Cannot iterate over elements with more than one dimension'
        aec = self.array_elemental_composition.reshape(-1, len(self.fs.elements))
        return iter({s: c for s, c in zip(self.fs.elements, aec[i, :])} for i in range(aec.shape[0]))

    def __repr__(self) -> str:
        if len(self.array_elemental_composition.shape) == 1:
            lines = [f'{s:16} {float(c):5.3e} mol' for s, c in zip(self.fs.elements, self.array_elemental_composition)]
            return '\n'.join(lines)
        else:
            array_disp = self.array_elemental_composition.__repr__()
            padding = int(array_disp.find('\n') / (len(self.fs.elements) + 1))
            return ('Elements:\n' + ' ' * int(padding / 2) +
                    ''.join([(' ' * (padding - len(s))) + s for s in self.fs.elements]) +
                    '\nMols:\n' + self.array_elemental_composition.__repr__())


def lookup(prop_array: FloatArray,
           temperature: FloatArray | float,
           t_offset: float,
           np: Any = np) -> FloatArray:
    """linear interpolates values from the given prop_array

    Args:
        prop_array: Array of the temperature depended property
        temperature: Absolute temperature(s) in Kelvin. Must
            be broadcastable to prop_array.
        np: Array namespace of prop_array

    Returns:
        Interpolates values based on given temperature
    """
    t = np.asarray(temperature) - t_offset
    t_lim = np.minimum(np.maximum(0, t), prop_array.shape[0] - 2)

    f = np.expand_dims(t - np.floor(t_lim), axis=-1)

    ti1 = t_lim.astype(int)
    return f * prop_array[ti1 + 1, :] + (1 - f) * prop_array[ti1, :]  # type: ignore[no-any-return]


def reorder_array(arr: FloatArray, old_index: list[str], new_index: list[str], np: Any = np) -> FloatArray:
    """Reorder the last dimension of an array according to a provided list of species
    names in the old oder and a list in the new order.

    Args:
        arr: Array to be reordered
        old_index: List of species names in the current order
        new_index: List of species names in the new order
        np: Array namespace of arr

    Returns:
        Array with the last dimension reordered
    """
    missing = [k for k in old_index if k not in new_index]
    if missing:
        raise ValueError(f'{missing[0]} is not in the new index')

    # Entries not available in the old index are taken from an appended column of zeros
    arr = np.concatenate([arr, np.zeros_like(arr[..., :1])], axis=-1)
    return arr[..., [old_index.index(k) if k in old_index else len(old_index) for k in new_index]]


@overload
def array_operation(self: elements, other: elements | fluid, func: Callable[[FloatArray, FloatArray], FloatArray]) -> elements:
    pass


@overload
def array_operation(self: fluid, other: T, func: Callable[[FloatArray, FloatArray], FloatArray]) -> T:
    pass


@overload
def array_operation(self: T, other: fluid, func: Callable[[FloatArray, FloatArray], FloatArray]) -> T:
    pass


def array_operation(self: elements | fluid, other: elements | fluid, func: Callable[[FloatArray, FloatArray], FloatArray]) -> elements | fluid:
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
            el_array = reorder_array(other.array_elemental_composition, other.fs.elements, self.fs.elements)
            return elements(func(self.array_elemental_composition, el_array), self.fs)
        else:
            el_array = reorder_array(other.array_composition, other.fs.species, self.fs.species)
            return fluid(func(self.array_composition, el_array), self.fs)
    elif set(self.fs.species) < set(other.fs.species):
        if isinstance(self, elements) or isinstance(other, elements):
            el_array = reorder_array(self.array_elemental_composition, self.fs.elements, other.fs.elements)
            return elements(func(el_array, other.array_elemental_composition), other.fs)
        else:
            el_array = reorder_array(self.array_composition, self.fs.species, other.fs.species)
            return fluid(func(el_array, other.array_composition), other.fs)
    else:
        new_fs = fluid_system(sorted(list(set(self.fs.species) | set(other.fs.species))),
                              backend=self.fs.np, device=self.fs.device)
        if isinstance(self, elements) or isinstance(other, elements):
            el_array1 = reorder_array(self.array_elemental_composition, self.fs.elements, new_fs.elements, self.fs.np)
            el_array2 = reorder_array(other.array_elemental_composition, other.fs.elements, new_fs.elements, other.fs.np)
            return elements(func(el_array1, el_array2), new_fs)
        else:
            el_array1 = reorder_array(self.array_composition, self.fs.species, new_fs.species, self.fs.np)
            el_array2 = reorder_array(other.array_composition, other.fs.species, new_fs.species, other.fs.np)
            return fluid(func(el_array1, el_array2), new_fs)
