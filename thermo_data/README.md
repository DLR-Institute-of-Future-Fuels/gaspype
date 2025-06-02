# Data preparation
Gaspype uses a binary format for thermodynamic data to minimize memory usage and startup time.

## Combine and prepare data
To prepare the data from YAML and XML sources:

``` bash
python combine_data.py combined_data.yaml nasa9*.yaml nasa9*.xml
```
General Syntax is: ```python combine_data.py OUTPUT_YAML_FILE INPUT_FILE_PATTERN_1 INPUT_FILE_PATTERN_2 ...```

Example output format for a single file entry:
``` yaml
- name: HCl
  composition: {Cl: 1, H: 1}
  thermo:
    model: NASA9
    temperature-ranges: [200.0, 1000.0, 6000.0]
    data:
    - [20625.88287, -309.3368855, 5.27541885, -0.00482887422, 6.1957946e-06, -3.040023782e-09, 4.91679003e-13, -10677.82299, -7.309305408]
    - [915774.951, -2770.550211, 5.97353979, -0.000362981006, 4.73552919e-08, 2.810262054e-12, -6.65610422e-16, 5674.95805, -16.42825822]
    note: Gurvich,1989 pt1 p186 pt2 p93. [tpis89]
```

Compile ```combined_data.yaml``` to the binary gaspype format:
``` bash
python compile_to_bin.py combined_data.yaml ../src/gaspype/data/therm_data.bin
```
General syntax is: ```python compile_to_bin.py YAML_INPUT_FILE BINARY_OUTPUT_FILE```

The binary format is structured like this, it uses little-endian and IEEE 754 floats:
```
[4 Byte magic number: 'gapy']
[8 Byte: 32 Bit integer for length of all species names (NAMES_LENGTH)]
[NAMES_LENGTH Bytes: ASCII encoded string with all species names separated by space]
[Index
    [For each species
        [4 Bytes: 32 Bit uint with offset of species data in file]
        [1 Byte:   8 Bit uint with number of elements]
        [1 Byte:   8 Bit uint for polygon length, value = 9]
        [1 Byte:   8 Bit uint for number of temperature supporting points (NUM_TEMPS)]
        [1 Byte:   8 Bit uint for length of reference string (REF_LEN)]
    ]
]
[Data
    [For each species
        [For each species element
            [2 Byte: element name in ASCII, 0x20 padded]
            [1 Byte: 8 Bit uint for number of atoms]
        ]
        [For Range(NUM_TEMPS)
            [4 Byte: 32 Bit float with temperature supporting point]
        ]
        [For Range(NUM_TEMPS - 1)
            [36 Bytes: 9 x 32 Bit float with NASA9-Polynomial for a temperature interval]
        ]
        [REF_LEN Bytes: ASCII string of the data reference]
    ]
]
```

## Notes
- Original source of the data compilation: https://ntrs.nasa.gov/citations/20020085330
- nasa9_*.yaml files are exported from https://cearun.grc.nasa.gov/ThermoBuild/ and
  converted with ck2yaml (https://cantera.org/stable/userguide/thermobuild.html)
- nasa9polynomials.xml is from: https://github.com/guillemborrell/thermopy/tree/master/databases