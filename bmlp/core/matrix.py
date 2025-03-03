from .utils import *
import time
import numpy as np
import graphblas as gb
from graphblas import Matrix, Vector, Scalar
from graphblas import dtypes
from graphblas import unary, binary, monoid, semiring

# Use python wrapper of GraphBLAS on GPU (BLAS - Basic Linear Algebra Subprograms)
# GraphBLAS supports graph operations via linear algebraic methods (e.g. matrix multiplication) over various semirings
# Documentation: https://graphegon.github.io/pygraphblas/pygraphblas/index.html

# Operator	Description	                    GraphBLAS Type
# A @ B	    Matrix Matrix Multiplication	type default PLUS_TIMES semiring
# v @ A	    Vector Matrix Multiplication	type default PLUS_TIMES semiring
# A @ v	    Matrix Vector Multiplication	type default PLUS_TIMES semiring
# A @= B	In-place Matrix Matrix Multiplication	type default PLUS_TIMES semiring
# v @= A	In-place Vector Matrix Multiplication	type default PLUS_TIMES semiring
# A @= v	In-place Matrix Vector Multiplication	type default PLUS_TIMES semiring
# A | B	    Matrix Union	type default SECOND combiner
# A |= B	In-place Matrix Union	type default SECOND combiner
# A & B	    Matrix Intersection	type default SECOND combiner
# A &= B	In-place Matrix Intersection	type default SECOND combiner
# The element-wise union performs the correct boolean operation on elements (None or BOOL)
# A + B	    Matrix Element-Wise Union	type default PLUS combiner
# A += B	In-place Matrix Element-Wise Union	type default PLUS combiner
# A - B	    Matrix Element-Wise Union	type default MINUS combiner
# A -= B	In-place Matrix Element-Wise Union	type default MINUS combiner
# The element-wise intersection performs the correct boolean operation on elements (None or BOOL) with higher operator order
# A * B	    Matrix Element-Wise Intersection	type default TIMES combiner
# A *= B	In-place Matrix Element-Wise Intersection	type default TIMES combiner
# A / B	    Matrix Element-Wise Intersection	type default DIV combiner
# A /= B	In-place Matrix Element-Wise Intersection	type default DIV combiner
# A == B	Compare Element-Wise Union	type default EQ operator
# A != B	Compare Element-Wise Union	type default NE operator
# A < B	    Compare Element-Wise Union	type default LT operator
# A > B	    Compare Element-Wise Union	type default GT operator
# A <= B	Compare Element-Wise Union	type default LE operator
# A >= B	Compare Element-Wise Union	type default GE operator


# Multiply two sparse matrices
def mul(M1: Matrix, M2: Matrix):
    return M1 @ M2


# Add two sparse matrices
def add(M1: Matrix, M2: Matrix):
    return M1 + M2


# Transpose a sparse matrix
def transpose(M: Matrix):
    return M.T


# Add an identity matrix to a square sparse matrix
def addI(M: Matrix):
    assert M.nrows == M.ncols
    return M + identity(M.nrows)


# Negate elements in a sparse matrix by treating empty cells as 'False'
def negate(M: Matrix):

    # Create an iso mask of default 'False' values
    OR = Matrix.iso(False, M.nrows, M.ncols)
    # Negate the OR results between the union of elements and select those with 'True'
    M = (M + OR).apply(types.BOOL.LNOT).select('==', True)
    return M


# Create an identity matrix
def identity(dim):
    I = Matrix(BOOL, dim, dim)
    for i in range(dim):
        I[i, i] = True
    return I


# Create a new matrix
def new_matrix(nrows, ncols, dtype=gb.dtypes.BOOL):
    return Matrix.new(dtype, nrows, ncols)


def RMS(P1, P2=None, print_matrix=False):
    """GraphBLAS version of BMLP-RMS algorithm which performs repeated matrix squaring

        P0, P1 and P2 are boolean matrices representing the following predicates.
        p0(X,Z):- p1(X,Z).
        p0(X,Z):- p2(X,Y),p0(Y,Z).

        Default p1 and p2 represent the same predicate.

    Args:
        P1 (Matrix): boolean matrix for the non recursive body p1 or default both p1 and p2
        P2 (Matrix, optional): boolean matrix for the recursive body p2 if differs from p1. Defaults to None.
        print_matrix (bool, optional): print trace of fixpoint computation. Defaults to False.

    Returns:
        Matrix: fixpoint boolean matrix representing predicate p0
    """

    # the dimensions of the matrix, e.g. the number of nodes in a graph
    dim = P1.nrows
    empty_matrix = Matrix(BOOL, dim, dim)

    # Add identy to the adjacency matrix
    R = identity(dim) + P1 if P2 is None else identity(dim) + P2
    if print_matrix:
        print('R = R2 + I = \n' + str(R) + '\n')

    # Iteratively compute the transitive closure using boolean matrix multiplication
    # Initialise closure matrix with an empty matrix
    R_ = empty_matrix
    while True:
        # R = R x R until no new connections are found
        R_ = R @ R
        if print_matrix:
            print('fixpoint = \n' + str(R_) + '\n')
        if R_.iseq(R):
            break
        R = R_

    # Multiply to remove redundant diagonal elements
    res = R_ @ P1

    if print_matrix:
        print('R0* = \n' + str(res) + '\n')

    return res


# GraphBLAS version of BMLP-SMP algorithm which performs vector multiplication
def SMP(V, R1, print_matrix=False):

    # Push the model subset selection into the summation for improved performance
    V_ = V
    while True:
        # Apply vector multiplication (selection) to the transitive closure
        V_ = V + V @ R1
        if print_matrix:
            print('V* = \n' + str(V_) + '\n')
        if V_.iseq(V):
            break
        V = V_

    # Multiply to remove redundant diagonal elements
    res = V_ @ R1

    if print_matrix:
        print('V* = \n' + str(res) + '\n')

    # Multiple to remove redundant elements
    return res


def IE(V: Matrix, R1: Matrix, R2: Matrix, T: Matrix = None,
       localised: bool = False, print_matrix: bool = False):
    """GraphBLAS version of BMLP-IE algorithm
        which performs matrix operations using two input matrices
        of dimension k x n.

        The input V is u rows of 1 x n vectors (a batch of u inputs).

    Args:
        V (Matrix): A matrix containing u 1 x n vectors.
        R1 (Matrix): A k x n boolean matrix.
        R2 (Matrix): A k x n boolean matrix.
        T (Matrix, optional): A u x k filter applied on R2. Defaults to None.
        localised (bool, optional): Use matrices stored in SuiteSparse specific binary file at R1 and R2 locations.
        print_matrix (bool, optional): Print trace of fixpoint computation. Defaults to False.

    Returns:
        Matrix: V* transitive closure (fixpoint) containing u rows of 1 x n vectors.
    """

    # Localised R1 is assumed transposed to avoid redundant operations
    if localised:
        R1 = Matrix.from_binfile(R1)
        R2 = Matrix.from_binfile(R2)
        nrows = R1.ncols
        ncols = max(R1.nrows, R2.ncols)
    # If R1 and R2 are not stored as SuiteSparse binary format
    # then R1 and R2 will be pre-processed as they have the same sizes
    # Otherwise, stored R1 has already been transposed and has a different size.
    else:
        nrows = R1.nrows
        ncols = max(R1.ncols, R2.ncols)
        R1.resize(nrows, ncols)
        R1 = R1.T
        R2.resize(nrows, ncols)

    # Need to resize matrices to be compatible due to leading zeros
    # Then Pad the input matrix with default 'False' values
    ninputs = V.nrows
    V.resize(ninputs, ncols)

    # When a filter on R2 is applied, negate it to find which R2 rows to keep
    if T is not None:
        T.resize(ninputs, nrows)
        T = T.apply(types.BOOL.LNOT)

    SNum = 0

    total_time = 0
    while True:
        # Find all rows that are subsets of V
        # Would need R2[i,j] -> R1[i,j] if matrix-matrix mul
        # can be performed on the element union

        # A sparse matrix workaround
        # Negate the V matrix while keeping it sparse to find all R1 rows that are not subsets
        # This can be done with sparse matrix mul without require union matrix-matrix mul
        # start_time = time.time()
        V_ = negate(V) @ R1
        V_ = negate(V_)
        # total_time += time.time() - start_time

        # Multiply with rows in R2 filtered by T and update
        if T is None:
            res = V_ @ R2 + V
        else:
            start_time = time.time()
            res = (V_.union(T, add_op=types.BOOL.MIN)) @ R2 + V
            total_time += time.time() - start_time
            # R1.to_binfile("reactant_mat_bin")
            # R2.to_binfile("product_mat_bin")

        if print_matrix:
            print('V* = \n' + str(res) + '\n')
        if res.iseq(V):
            break
        V = res
        SNum += 1

    # print(total_time)

    if print_matrix:
        print('V* = \n' + str(res) + '\n')

    return res, SNum
