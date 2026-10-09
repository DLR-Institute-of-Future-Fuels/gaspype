# Carbon Activity

This example shows the calculation of the carbon activity for methane mixtures
in thermodynamic equilibrium.

```python
import gaspype as gp
import numpy as np
import matplotlib.pyplot as plt
```

Setting temperatures and pressure:


```python
t_range = np.array([600, 700, 800, 900, 1100, 1500])  # °C

p = 1e5  # Pa

fs = gp.fluid_system(['H2', 'H2O', 'CO2', 'CO', 'CH4'])
```

Equilibrium calculation for methane steam mixtures:


```python
ratio = np.linspace(0.01, 1.5, num=128)

fl = gp.fluid({'CH4': 1}, fs) + ratio * gp.fluid({'H2O': 1}, fs)
```

The shapes of the fluid and the temperature are broadcast against each other.
So we can calculate the carbon activity for all compositions times all
temperatures in t_range with a single call:


```python
carbon_activity = gp.carbon_activity(fl, t_range[:, None] + 273.15, p)
```

Plot carbon activities, a activity of > 1 means there is thermodynamically the formation of sold carbon favored.


```python
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
ax.set_xlabel("H2O/CH4")
ax.set_ylabel("carbon activity")
ax.set_ylim(1e-1, 1e3)
ax.set_yscale('log')
ax.plot(ratio, carbon_activity.T)
ax.hlines(1, np.min(ratio), np.max(ratio), colors='k', linestyles='dashed')
ax.legend([f'{tc} °C' for tc in t_range])
```

Let's do the equilibrium calculation for methane CO2 mixtures as well:


```python
fl_co2 = gp.fluid({'CH4': 1}, fs) + ratio * gp.fluid({'CO2': 1}, fs)
carbon_activity_co2 = gp.carbon_activity(fl_co2, t_range[:, None] + 273.15, p)
```

And plot carbon activities over the CO2 to CH4 ratio:


```python
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
ax.set_xlabel("CO2/CH4")
ax.set_ylabel("carbon activity")
ax.set_ylim(1e-1, 1e3)
ax.set_yscale('log')
ax.plot(ratio, carbon_activity_co2.T)
ax.hlines(1, np.min(ratio), np.max(ratio), colors='k', linestyles='dashed')
ax.legend([f'{tc} °C' for tc in t_range])
```

gaspype.carbon_activity is a shortcut for the more general function
gaspype.activity, which calculates the activity of any substance in the
equilibrated gas. Condensed species (solids and liquids) can be listed with
gaspype.species(condensed=True). For example the activity of liquid water
shows if steam condenses, here for a fuel gas at 50 °C:


```python
fl_wet = gp.fluid({'H2': 0.8, 'H2O': 0.2}, fs)

gp.activity(fl_wet, 50 + 273.15, p, 'H2O(L)')
```
