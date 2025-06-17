import struct
from typing import Generator, Iterator
from dataclasses import dataclass


def split_on_space(data: bytes, offset: int, end: int) -> Iterator[str]:
    """Splits a byte array into ASCII strings based on spaces.
    Args:
        data: The byte array to split.
        offset: The starting index in the byte array.
        end: The ending index in the byte array.
    Yields:
        str: ASCII strings found in the byte array.
    """
    start = offset
    for i in range(offset, end):
        if data[i] == 0x20:  # ASCII space character
            if start < i:
                yield data[start:i].decode('ascii')
            start = i + 1
    if start < end:
        yield data[start:end].decode('ascii')


@dataclass
class SpeciesData:
    """Class to hold the physical data for a species.
    Attributes:
        name: Name of the species.
        composition: Dictionary of species composition with element symbols as keys and their counts as values.
        model: Number of polynomial coefficients used in the model.
        ref_string: Reference string for the data source.
        t_range: List of temperatures nodes marking intervals.
        data: List of lists containing physical data for each temperature interval.
    """
    name: str
    composition: dict[str, int]
    model: int
    ref_string: str
    t_range: list[float]
    data: list[list[float]]


class db_reader():
    """Class to read and parse the binary gas phase species database.

    Attributes:
        names: An iterator over the names of species in the database.
    """

    header_len = 8

    def __init__(self, inp_data: bytes):
        """ Initializes the database reader with binary data.
        Args:
            inp_data: The binary data of the gas phase species database.
        """
        assert inp_data[:4] == b'gapy', 'Unknown data format'
        self._bin_data = inp_data
        self._name_lengths = struct.unpack('<I', self._bin_data[4:8])[0]
        species_names = split_on_space(self._bin_data, db_reader.header_len, db_reader.header_len + self._name_lengths)
        self._index = {s: i for i, s in enumerate(species_names)}

    @property
    def names(self) -> Iterator[str]:
        return iter(self._index.keys())

    def __iter__(self) -> Generator[SpeciesData, None, None]:
        return (d for d in (self.read(species) for species in self._index.keys()) if d is not None)

    def __contains__(self, name: str) -> bool:
        """ Checks if a species name is present in the database.
        Args:
            name: The name of the species to check.

        Returns:
            bool: True if the species name is present
        """
        return name in self._index

    def __getitem__(self, name: str) -> SpeciesData:
        species_data = self.read(name)
        assert species_data, f"Species '{name}' not found in the database."
        return species_data

    def read(self, name: str) -> SpeciesData | None:
        """ Reads the physical data for a given species name from the binary data.
        Args:
            name (str): The name of the species to read data for.

        Returns:
            phys_data: An instance of the phys_data class containing the physical data.
        """
        if name not in self._index:
            return None

        head_offset = self._name_lengths + db_reader.header_len + self._index[name] * db_reader.header_len

        head = struct.unpack('<I4B', self._bin_data[head_offset:head_offset + db_reader.header_len])

        offset = head[0]
        composition_count = head[1]
        model = head[2]
        temperature_count = head[3]
        ref_string_len = head[4]

        td_data_num = (temperature_count - 1) * model
        data_len = composition_count * 3 + (temperature_count + td_data_num) * 4 + ref_string_len

        format_string = '<' + '2sB' * composition_count + f'{temperature_count}f{td_data_num}f{ref_string_len}s'

        bindat = struct.unpack(format_string, self._bin_data[offset:offset + data_len])
        comp = {bindat[i * 2].strip().decode('ASCII'): bindat[i * 2 + 1] for i in range(composition_count)}

        noffs = composition_count * 2
        t_range = list(bindat[noffs:noffs + temperature_count])

        noffs += temperature_count
        data = [list(bindat[(noffs + i * model):(noffs + (i + 1) * model)]) for i in range(temperature_count - 1)]
        ref = bindat[-1].decode('utf-8')

        return SpeciesData(name, comp, model, ref, t_range, data)


"""
Atomic weights values are used from CIAAW.
when a single value is given. Available online at
http://www.ciaaw.org/atomic-weights.htm
When a range of values is given in the CIAAW table, the "conventional
atomic weight" from the IUPAC Periodic Table is used. Available
online at https://iupac.org/wp-content/uploads/2018/12/IUPAC_Periodic_Table-01Dec18.pdf
"""
atomic_weights = {'H': 1.008, 'He': 4.002602, 'Li': 6.94, 'Be': 9.0121831, 'B': 10.81, 'C': 12.011,
                  'N': 14.007, 'O': 15.999, 'F': 18.998403163, 'Ne': 20.1797, 'Na': 22.98976928,
                  'Mg': 24.305, 'Al': 26.9815384, 'Si': 28.085, 'P': 30.973761998, 'S': 32.06,
                  'Cl': 35.45, 'Ar': 39.95, 'K': 39.0983, 'Ca': 40.078, 'Sc': 44.955908,
                  'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938043, 'Fe': 55.845,
                  'Co': 58.933194, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723,
                  'Ge': 72.63, 'As': 74.921595, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
                  'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90584, 'Zr': 91.224, 'Nb': 92.90637,
                  'Mo': 95.95, 'Ru': 101.07, 'Rh': 102.90549, 'Pd': 106.42, 'Ag': 107.8682,
                  'Cd': 112.414, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6,
                  'I': 126.90447, 'Xe': 131.293, 'Cs': 132.90545196, 'Ba': 137.327, 'La': 138.90547,
                  'Ce': 140.116, 'Pr': 140.90766, 'Nd': 144.242, 'Sm': 150.36, 'Eu': 151.964,
                  'Gd': 157.25, 'Tb': 158.925354, 'Dy': 162.5, 'Ho': 164.930328, 'Er': 167.259,
                  'Tm': 168.934218, 'Yb': 173.045, 'Lu': 174.9668, 'Hf': 178.49, 'Ta': 180.94788,
                  'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084,
                  'Au': 196.96657, 'Hg': 200.592, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.9804,
                  'Th': 232.0377, 'Pa': 231.03588, 'U': 238.02891}
