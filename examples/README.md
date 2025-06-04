# Example scripts

``` bash
notedown .\docs\source\examples\soec_methane.md --to notebook --output .\docs\files\soec_methane.ipynb --run

maybe: pip install ipykernel jupyter  
maybe: python -m ipykernel install --user --name temp_kernel --display-name "Python (temp_kernel)"


jupyter nbconvert --to markdown .\docs\files\soec_methane.ipynb --output .\docs\files\soec_methane_out.md
```

