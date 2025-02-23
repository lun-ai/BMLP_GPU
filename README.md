## Installation
BMLP-GPU is usable both from Python and SWI-Prolog.

### Conda installation
Download conda ([Anaconda](https://www.anaconda.com/download/)) for your device. Then run in terminal, e.g.
```
bash Anaconda-latest-Linux-x86_64.sh
conda list
```
To confirm installation success, you should see the list of installed conda packages. 

#### Dependency

BMLP-GPU requires the following packages in a conda environment:
- python
  - cugraph 
  - pygraphblas
  - cudf (optional for pandas with GPU)
  - ipykernel (optional for running Jupyter notebook)
  - graphviz (optional for visualising graphs)


To create a conda environment with above packages:
```
conda env create -f environment.yml
conda activate bmlp
```

## Examples
Two BMLP modules RMS and SMP [1] are in bmlp.py.

See the list of linear algebra operations supported by PyGraphBLAS in bmlp.py.

## Calling from Python
Please see examples for calling RMS and SMP modules in examples.ipynb.

## Calling from SWI-Prolog

As an example, the following clauses in the test/int_matrix.pl file describe rows in a boolean matrix, we show how to use BMLP-GPU to compute its transitive closure.
```
            0  1  2  3  4
a(2).   0|     1         |  0
a(4).   1|        1      |  1
a(8).   2|           1   |  2
a(1).   3|  1            |  3
a(16).  4|              1|  4
            0  1  2  3  4
```

On a terminal session, create a Python server for SWI-Prolog clients.
``` 
python -m bmlp.service.swipl_server
```
On another terminal, create a SWI-Prolog client (alternatively, you can consult swipl_client.pl in your source code).
```
swipl bmlp/service/swipl_client.pl
```

On the SWI-Prolog client side, request the conversion from integers in test/int_matrix.pl to a graphBLAS sparse matrix. The converted matrix is stored by the python variable 'a'.
```
?- run_python_command("import bmlp.core.utils", Res).
?- run_python_command("a = bmlp.core.utils.integers_to_boolean_matrix('test/int_matrix.pl')", Res).
```

Now, we call RMS to compute the transitive closure taking a single matrix argument. Here we input 'a'. The computed result is assigned to 'a' again.
```
?- run_python_command("import bmlp.core.matrix", Res).
?- run_python_command("a = bmlp.core.matrix.RMS(a)", Res).
```

The final step is to convert the graphBLAS sparse matrix back to SWI-Prolog. The first argument is a graphBLAS matrix, the second argument is the name of the new predicate, the third argument is the file location where the set of clauses will be saved into. 
```
?- run_python_command("a = bmlp.core.utils.boolean_matrix_to_integers(a,'a','test/output.pl')", Res).
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

## References

[1] Ai, Lun, and Stephen H. Muggleton. ‘Boolean Matrix Logic Programming’. arXiv, 19 August 2024. https://doi.org/10.48550/arXiv.2408.10369.