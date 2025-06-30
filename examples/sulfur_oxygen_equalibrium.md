# Sulfur Oxygen Equilibrium

This example shows equilibrium calculations for sulfur/oxygen mixtures.

```python
import gaspype as gp
import numpy as np
import matplotlib.pyplot as plt
```

List possible sulfur/oxygen species:

```python
gp.species(element_names = 'S, O')
```

Or more specific by using regular expressions:


```python
gp.species('S?[2-3]?O?[2-5]?', use_regex=True)
```

Calculation of the molar equilibrium fractions for sulfur and oxygen depending on the oxygen to sulfur ratio: 


```python
fs = gp.fluid_system(['S2', 'S2O', 'SO2', 'SO3', 'O2'])

oxygen_ratio = np.linspace(0.5, 3, num=128)
el = gp.elements({'S': 1}, fs) + oxygen_ratio * gp.elements({'O': 1}, fs)

composition = gp.equilibrium(el, 800+273.15, 1e4)

fig, ax = plt.subplots()
ax.set_xlabel("Oxygen to sulfur ratio")
ax.set_ylabel("Molar fraction")
ax.plot(oxygen_ratio, composition.get_x(), '-')
ax.legend(composition.species)
```

Calculation of the molar equilibrium fractions for sulfur and oxygen depending on temperature in °C:


```python
fs = gp.fluid_system(['S2', 'S2O', 'SO2', 'SO3', 'O2'])

el = gp.elements({'S': 1, 'O':2.5}, fs)

t_range = np.linspace(500, 1300, num=32)
composition = gp.equilibrium(el, t_range+273.15, 1e4)

fig, ax = plt.subplots()
ax.set_xlabel("Temperature / °C")
ax.set_ylabel("Molar fraction")
ax.plot(t_range, composition.get_x(), '-')
ax.legend(composition.species)
```
