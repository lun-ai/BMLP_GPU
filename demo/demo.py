import pygraphblas as gb
from bmlp.Matrix import *
from bmlp.Call import *
import time


def test_bmlp_ie_demo_bio_db():

    # Test BMLP-IE on BMLP_active datasets
    # Load datasets
    R1 = integers_to_boolean_matrix("demo/mstate1")
    R2 = integers_to_boolean_matrix("demo/mstate3")
    # Load inputs
    V = integers_to_boolean_matrix("demo/mstate5")
    T = integers_to_boolean_matrix("demo/mstate2")

    # Run BMLP-IE
    res, _ = BMLP_IE(V, R1, R2, T=T)

    # Save computed result to local
    boolean_matrix_to_integers(res, "mstate10", "demo/mstate10")


def test_bmlp_ie_demo_bio_db_from_bin():

    # Test BMLP-IE on BMLP_active datasets
    # Load datasets stored in SuiteSparse binary format
    R1 = "demo/reactant_mat_bin"
    R2 = "demo/product_mat_bin"
    # Load inputs
    V = integers_to_boolean_matrix("demo/mstate5")
    T = integers_to_boolean_matrix("demo/mstate2")

    # Run BMLP-IE
    res, _ = BMLP_IE(V, R1, R2, T=T, localised=True)

    # Save computed result to local
    boolean_matrix_to_integers(res, "mstate10", "demo/mstate10")


if __name__ == '__main__':
    # test_bmlp_ie_demo_bio_db()
    # test_bmlp_ie_demo_bio_db_from_bin()

    # integers_to_boolean_matrix("demo/mstate2_500")
    BMLP_IE_gpu("demo/reactant_mat_bin", "demo/product_mat_bin",
                "demo/mstate5_500", "demo/mstate2_500", "mstate10", "demo/mstate10")
