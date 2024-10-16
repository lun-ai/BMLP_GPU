## Installation
Follow the steps to install and run BMLP-GPU.

### Conda installation
Download conda ([Anaconda](https://www.anaconda.com/download/)) for your device. Then run in terminal, e.g.
```
bash Anaconda-latest-Linux-x86_64.sh
conda list
```
To confirm installation success, you should see the list of install conda packages. 

#### Dependency

BMLP-GPU requires the following packages to be installed in a conda environment:
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

## Calling from SWI-Prolog

As an example, the following clauses in the test.pl file describe rows in a boolean matrix, we show how to use BMLP-GPU to compute its transitive closure.
```
            0  1  2  3  4
a(2).   0|     1         |  0
a(4).   1|        1      |  1
a(8).   2|           1   |  2
a(1).   3|  1            |  3
a(16).  4|              1|  4
            0  1  2  3  4
```

On one terminal, create a Python server for SWI-Prolog clients.
``` 
python -m bmlp.swipl_server
```
On another terminal, create a SWI-Prolog client.
```
swipl bmlp/swipl_client.pl
```
Note: alternatively, you can use the swipl_client.pl module in your source code.

Then, on the SWI-Prolog Client side, request the conversion from integers in test.pl to a graphBLAS sparse matrix. The converted matrix is stored by the python variable 'a'.
```
?- run_python_command(
  "a = bmlp.matrix.integers_to_boolean_matrix('test. pl')", Res).
```

Now, we call BMLP_RMS to compute the transitive closure taking a single matrix argument. Here we input 'a'. The computed result is assigned to 'a' again.
```
?- run_python_command("a = bmlp.matrix.BMLP_RMS(a)", Res).
```

The final step is to convert the graphBLAS sparse matrix back to SWI-Prolog. The first argument is a graphBLAS matrix, the second argument is the name of the new predicate, the third argument is the file location where the set of clauses will be saved into. 
```
?- run_python_command(
  "a = bmlp.matrix.boolean_matrix_to_integers(a,'a','output.pl')", 
  Res).
```

The output.pl file now contains clauses that represent the transitive closure boolean matrix.
```
            0  1  2  3  4
a(15).  0|  1  1  1  1   |  0
a(15).  1|  1  1  1  1   |  1
a(15).  2|  1  1  1  1   |  2
a(15).  3|  1  1  1  1   |  3
a(16).  4|              1|  4
            0  1  2  3  4
```
  


## Source code
Two modules BMLP-RMS and BMLP-SMP are implemented in bmlp.py.

See the list of linear algebra operations supported by PyGraphBLAS in bmlp.py.
