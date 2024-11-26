from .Matrix import *


# V and T are the input matrices, e.g. metabolite medium and gene knockout conditions
def BMLP_IE_gpu(R1_path, R2_path, V, T, P_name='', Res_path=''):

    # Load datasets stored in SuiteSparse binary format
    R1 = R1_path
    R2 = R2_path

    # Load inputs
    if P_name != '' and Res_path != '':
        V_m = integers_to_boolean_matrix(V)
        T_m = integers_to_boolean_matrix(T)

        # Run BMLP-IE
        res, _ = BMLP_IE(V_m, R1, R2, T=T_m, localised=True)

        # Save computed result to result path
        boolean_matrix_to_integers(res, P_name, Res_path)
        return None
    else:
        # Run BMLP-IE
        res, _ = BMLP_IE(V, R1, R2, T=T, localised=True)
        return res
