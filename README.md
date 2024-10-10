## Dependency
- conda
- python
  - cudf
  - cugraph 
  - pygraphblas
  - ipykernel (optional for running Jupyter notebook)
  - graphviz (optional for visualising graphs)

To create a conda environment with above packages:
```
conda env create -f environment.yml
conda activate gpu-env
```

## Source code
Two modules BMLP-RMS and BMLP-SMP are implemented in bmlp.py.

See the list of linear algebra operations supported by PyGraphBLAS in bmlp.py.

