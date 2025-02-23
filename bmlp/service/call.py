from ..core.matrix import *
from ..core.utils import *


# V and T are the input matrices, e.g. metabolite medium and gene knockout conditions
def IE_from_bin(R1_path, R2_path, V, T, P_name='', Res_path=''):

    # Load datasets stored in SuiteSparse binary format
    R1 = R1_path
    R2 = R2_path

    # Load inputs from .pl file containing rows of integers encoding of matrices
    if P_name != '' and Res_path != '':
        V_m = integers_to_boolean_matrix(V)
        T_m = integers_to_boolean_matrix(T)

        # Run BMLP-IE
        res, _ = IE(V_m, R1, R2, T=T_m, localised=True)

        # Save computed result to result path
        boolean_matrix_to_integers(res, P_name, Res_path)
        return None
    else:
        # Run BMLP-IE
        res, _ = IE(V, R1, R2, T=T, localised=True)
        return res
