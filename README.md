# Gaspype
Gaspype is a performant Python library for thermodynamic calculations like equilibrium
reactions for several hundred gas species and their mixtures - written in Python/NumPy,
with GPU support with JAX.

It is designed to address the needs of researchers and engineers working in chemical
engineering, combustion analysis, and energy systems. Thermodynamic calculations,
especially equilibrium reactions in gas mixtures, are essential for understanding
processes such as fuel combustion, solid oxide cell (SOFC/SOEC) operations, and other
high-temperature chemical reactions. Many existing tools present barriers to entry,
whether due to limited software development experience or restrictive licensing of
closed-source packages. 

This library aims to minimize friction by providing a high-level abstraction and a
ergonomic API, making it accessible for both rapid exploratory calculations and
integration into large-scale models. Gaspype is implemented in pure Python, fully
typed, and leverages NumPy vectorization to combine high performance with an
intuitive interface. It was developed based on practical experience with
spatially-resolved modeling of solid oxide cells and high-temperature solar
applications, ensuring its suitability for a wide range of thermodynamic modeling
tasks.

Compared to other open-source packages like Cantera, Gaspype offers a streamlined,
Pythonic API and competitive performance, despite being implemented entirely in
Python. Its open-source nature and minimal dependencies make it an accessible and
powerful tool for researchers in chemistry, chemical engineering, energy systems,
and electrochemistry.

Besides NumPy, [JAX](https://docs.jax.dev) can be selected as array backend. This
allows running calculations on a GPU, compiling them just-in-time and differentiating
them automatically.

## Key Features
- Pure Python implementation with NumPy vectorization for high performance
- Immutable types and comprehensive type hints for reliability
- Intuitive, Pythonic API for both rapid prototyping and complex multidimensional models
- Ready for Jupyter Notebook and educational use
- Optional JAX backend for GPU support, just-in-time compilation and automatic differentiation
- Ships with a comprehensive NASA9-based species database (500+ species with NASA9-polynomials)
- Supports electrochemical calculations including Nernst potentials and cell voltages
- Optimized binary database format for fast species lookup and minimal memory usage

## Installation
Installation with pip:
``` bash
pip install gaspype
```

Installation with conda:
``` bash
conda install conda-forge::gaspype
```

NumPy is the only required dependency. For the optional JAX backend install
Gaspype together with JAX:
``` bash
pip install gaspype[jax]
```
This installs the CPU version of JAX. For running calculations on a GPU follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Getting started
Gaspype provides two main classes: ```fluid``` and ```elements```.

### Fluid
A fluid class describes a mixture of molecular species and their individual molar amounts.

``` python
import gaspype as gp
fl = gp.fluid({'H2O': 1, 'H2': 2})
fl
```
```
Total            3.000e+00 mol
H2O              33.33 %
H2               66.67 %
```

Its functions provide thermodynamic, mass balance and ideal gas properties of the mixture.

``` python
cp = fl.get_cp(t=800+273.15)
mass = fl.get_mass()
gas_volume = fl.get_v(t=800+273.15, p=1e5)
```

The arguments can be provided as NumPy-arrays:

``` python
import numpy as np
t_range = np.linspace(600, 800, 5) + 273.15
fl.get_density(t=t_range, p=1e5)
```
```
array([0.10122906, 0.09574625, 0.09082685, 0.08638827, 0.08236328])
```
A ```fluid``` object can have multiple compositions. A multidimensional ```fluid``` object
can be created for example by multiplication with a NumPy array:

``` python
fl2 = gp.fluid({'H2O': 1, 'N2': 2}) + \
      np.linspace(0, 10, 4) * gp.fluid({'H2': 1})
fl2
```
```
Total mol:
array([ 3.        ,  6.33333333,  9.66666667, 13.        ])
Species:
              H2        H2O         N2
Molar fractions:
array([[0.        , 0.33333333, 0.66666667],
       [0.52631579, 0.15789474, 0.31578947],
       [0.68965517, 0.10344828, 0.20689655],
       [0.76923077, 0.07692308, 0.15384615]])
```
A fluid object can be converted to a pandas dataframe:
``` python
import pandas as pd
pd.DataFrame(list(fl2))
```
|    | H2O | N2  |  H2 
|----|-----|-----|-------
|0   | 1.0 | 2.0 | 0.000000
|1   | 1.0 | 2.0 | 3.333333
|2   | 1.0 | 2.0 | 6.666667
|3   | 1.0 | 2.0 | 10.000000

The broadcasting behavior is not limited to 1D-arrays:

``` python
fl3 = gp.fluid({'H2O': 1}) + \
      np.linspace(0, 10, 4) * gp.fluid({'H2': 1}) + \
      np.expand_dims(np.linspace(1, 3, 3), axis=1) * gp.fluid({'N2': 1})
fl3
```
```
Total mol:
array([[ 2.        ,  5.33333333,  8.66666667, 12.        ],
       [ 3.        ,  6.33333333,  9.66666667, 13.        ],
       [ 4.        ,  7.33333333, 10.66666667, 14.        ]])
Species:
              H2        H2O         N2
Molar fractions:
array([[[0.        , 0.5       , 0.5       ],
        [0.625     , 0.1875    , 0.1875    ],
        [0.76923077, 0.11538462, 0.11538462],
        [0.83333333, 0.08333333, 0.08333333]],

       [[0.        , 0.33333333, 0.66666667],
        [0.52631579, 0.15789474, 0.31578947],
        [0.68965517, 0.10344828, 0.20689655],
        [0.76923077, 0.07692308, 0.15384615]],

       [[0.        , 0.25      , 0.75      ],
        [0.45454545, 0.13636364, 0.40909091],
        [0.625     , 0.09375   , 0.28125   ],
        [0.71428571, 0.07142857, 0.21428571]]])
```

### Elements
In some cases not the molecular but the atomic composition is of interest.
The ```elements``` class can be used for atom based balances and works similar:

``` python
el = gp.elements({'N': 1, 'Cl': 2})
el.get_mass()
```
```
np.float64(0.08490700000000001)
```
A ```elements``` object can be as well instantiated from a ```fluid``` object.
Arithmetic operations between ```elements``` and ```fluid``` result in
an ```elements``` object:
``` python
el2 = gp.elements(fl) + el - 0.3 * fl
el2
```
```
Cl               2.000e+00 mol
H                4.200e+00 mol
N                1.000e+00 mol
O                7.000e-01 mol
```

Going from an atomic composition to a molecular composition is possible as well.
One way is to calculate the thermodynamic equilibrium for a mixture:

``` python
fs = gp.fluid_system('CH4, H2, CO, CO2, O2')
el3 = gp.elements({'C': 1, 'H': 2, 'O':1}, fs)
fl3 = gp.equilibrium(el3, t=800)
fl3
```
```
Total            1.204e+00 mol
CH4              33.07 %
H2               16.93 %
CO               16.93 %
CO2              33.07 %
O2                0.00 %
```

The ```equilibrium``` function can be called with a ```fluid``` or ```elements``` object
as first argument. ```fluid``` and ```elements``` referencing a ```fluid_system``` object
which can be set as shown above during the object instantiation. If not provided,
a new one will be created automatically. Providing a ```fluid_system``` gives more
control over which molecular species are included in derived ```fluid``` objects.
Furthermore arithmetic operations between objects with the same ```fluid_system```
are potentially faster:

``` python
fl3 + gp.fluid({'CH4': 1}, fs)
```
```
Total            2.204e+00 mol
CH4              63.44 %
H2                9.24 %
CO                9.24 %
CO2              18.07 %
O2                0.00 %
```

Especially if the ```fluid_system``` of one of the operands has not a subset of
molecular species of the other ```fluid_system``` a new ```fluid_system``` will
be created for the operation which might degrade performance:

``` python
fl3 + gp.fluid({'NH3': 1})
```
```
Total            2.204e+00 mol
CH4              18.07 %
CO                9.24 %
CO2              18.07 %
H2                9.24 %
NH3              45.38 %
O2                0.00 %
```

## JAX backend
The array library is selected per ```fluid_system``` with the ```backend``` argument.
All ```fluid``` and ```elements``` objects based on it use JAX for their arrays
and calculations and return JAX arrays. Arguments can still be provided as
floats or NumPy arrays:

``` py
import gaspype as gp
import numpy as np
import jax

jax.config.update('jax_enable_x64', True)

fs = gp.fluid_system('CH4, H2O, H2, CO, CO2', backend='jax')
fl = gp.fluid({'CH4': 1, 'H2O': 2}, fs)

t_range = np.linspace(600, 800, 5) + 273.15
fl.get_density(t=t_range, p=1e5)
```
```
Array([0.23909379, 0.2261439 , 0.21452473, 0.20404119, 0.19453454],      dtype=float64)
```

JAX calculates by default with single precision (float32). Gaspype works with
single precision as well, but with reduced accuracy for example for trace species
in an equilibrium. Double precision is enabled as shown above
with ```jax_enable_x64```.

A device can be selected with the ```device``` argument, either as string
like ```'cpu'```, ```'gpu'``` or ```'gpu:1'``` or as JAX device object:

``` py
fs_gpu = gp.fluid_system('CH4, H2O, H2, CO, CO2', backend='jax', device='gpu')
```

Gaspype functions can be used in JAX transformations like ```jax.jit```
and ```jax.vmap```:

``` py
@jax.jit
def h2_fraction(t):
    return gp.equilibrium(fl, t, p=1e5).get_x('H2')

h2_fraction(t_range)
```
```
Array([0.53103609, 0.59349886, 0.62796147, 0.63916901, 0.63958296],      dtype=float64)
```

They can be differentiated automatically as well, for example to get the change
of the equilibrium composition with the temperature:

``` py
jax.jacfwd(h2_fraction)(1000.5)
```
```
Array(0.00019654, dtype=float64)
```

Notes on the JAX backend:
- A ```fluid_system``` created automatically by an operation between two objects
  inherits the backend and device of the first operand. Objects created without
  a ```fluid_system``` use NumPy, therefore it is recommended to provide
  the ```fluid_system``` explicitly when using JAX.
- Compiled or vectorized equilibrium calculations can only be differentiated
  in forward mode (```jax.jacfwd```, ```jax.jvp```). Reverse mode (```jax.grad```,
  ```jax.jacrev```) works only without ```jax.jit``` and ```jax.vmap```.


## Developer Guide
Contributions are welcome, please open an issue or submit a pull request on GitHub.

To get started with developing the `gaspype` package, follow these steps.

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/DLR-Institute-of-Future-Fuels/gaspype.git
cd gaspype
```

It's recommended to setup an venv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

Install the package and dev-dependencies while keeping the package files
in the current directory:

```bash
pip install -e .[dev]
```

Compile binary property database from text based files:

```bash
python thermo_data/combine_data.py thermo_data/combined_data.yaml thermo_data/nasa9*.yaml thermo_data/nasa9*.xml
python thermo_data/compile_to_bin.py thermo_data/combined_data.yaml src/gaspype/data/therm_data.bin
python thermo_data/combine_data.py --condensed thermo_data/combined_data_condensed.yaml thermo_data/nasa9*.xml
python thermo_data/compile_to_bin.py --double thermo_data/combined_data_condensed.yaml src/gaspype/data/therm_data_condensed.bin
```

Ensure that everything is set up correctly by running the tests:

```bash
pytest
```

## Limitations
- **Ideal gas assumption**: Gaspype treats species as ideal gases, limiting applicability to moderate pressures or high-temperature applications.
- **Isobaric equilibrium**: Currently, only isobaric (constant pressure) equilibrium calculations are implemented.

## Quality Assurance
Gaspype's calculations are validated against reference data from:
- **Refprop** - for thermodynamic properties
- **Cantera** - for equilibrium calculations
- **Cycle-Tempo** - for additional equilibrium validation

The test suite includes over 1,000 reference values and covers all code snippets from the documentation.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
