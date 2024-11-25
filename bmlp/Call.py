from .Matrix import *

# V and T are the input matrices, e.g. metabolite medium and gene knockout conditions


def BMLP_IE_gpu(R1_path, R2_path, V_path, T_path, P_name, Res_path):

    # Load datasets stored in SuiteSparse binary format
    R1 = R1_path
    R2 = R2_path

    # Load inputs
    V = integers_to_boolean_matrix(V_path)
    T = integers_to_boolean_matrix(T_path)

    # Run BMLP-IE
    res, _ = BMLP_IE(V, R1, R2, T=T, localised=True)

    # Save computed result to result path
    boolean_matrix_to_integers(res, P_name, Res_path)


# V and T are the input matrices, e.g. metabolite medium and gene knockout conditions
def BMLP_IE_gpu(R1_path, R2_path, V, T):

    # Load datasets stored in SuiteSparse binary format
    R1 = R1_path
    R2 = R2_path

    # Run BMLP-IE
    res, _ = BMLP_IE(V, R1, R2, T=T, localised=True)

    return res
