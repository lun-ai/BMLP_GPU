from .utils import *
import numpy as np
import graphblas as gb
from graphblas import Matrix, Vector
from graphblas import dtypes as types
import pandas as pd
from tqdm import tqdm

# Use python wrapper of GraphBLAS on GPU (BLAS - Basic Linear Algebra Subprograms)
# GraphBLAS supports graph operations via linear algebraic methods (e.g. matrix multiplication) over various semirings
# Documentation: https://python-graphblas.readthedocs.io/en/stable/index.html


# Save and load a matrix from a DataFrame
def load(file_path):
    df = pd.read_csv(file_path)
    return Matrix.from_coo(df['row'].iloc[1:].values, df['col'].iloc[1:].values, True, nrows=df['row'].iloc[0], ncols=df['col'].iloc[0])


def save(matrix, file_path):
    coo = matrix.to_coo()

    df = pd.DataFrame({
        'attribute': ['dim'] + ['elem'] * len(coo[0]),
        'row': np.insert(coo[0], 0, matrix.nrows),
        'col': np.insert(coo[1], 0, matrix.ncols)
    })

    df.to_csv(file_path, index=False)


# Multiply two sparse matrices
# The default semiring for boolean is land_lor (logical AND and logical OR)
def mul(M1: Matrix, M2: Matrix, inplace=False):
    if inplace:
        return gb.semiring.land_lor(M1 @ M2)
    return gb.semiring.land_lor(M1 @ M2).new()


def square(M: Matrix, inplace=False):
    if inplace:
        return gb.semiring.land_lor(M @ M)
    return gb.semiring.land_lor(M @ M).new()


# Add two sparse matrices
def add(M1: Matrix, M2: Matrix, inplace=False):
    if inplace:
        return gb.binary.lor(M1 | M2)
    return gb.binary.lor(M1 | M2).new()


# Find the intersection between two sparse matrices
def intersection(M1: Matrix, M2: Matrix, inplace=False):
    if inplace:
        return gb.binary.land(M1 & M2)
    return gb.binary.land(M1 & M2).new()


# Transpose a sparse matrix
def transpose(M: Matrix, inplace=False):
    if inplace:
        return M.T
    return M.T.new()


# Add an identity matrix to a square sparse matrix
def addI(M: Matrix, inplace=False):
    assert M.nrows == M.ncols
    return add(M, identity(M.nrows), inplace)


# Negate elements in a sparse matrix by treating empty cells as 'False'
def negate(M: Matrix, inplace=False):
    # Create an iso mask of default 'False' values
    OR = Matrix.from_scalar(False, M.nrows, M.ncols)
    # Negate the OR results between the union of elements and select those with 'True'
    if inplace:
        return gb.unary.lnot(M | OR).select('==', True)
    return gb.unary.lnot(M | OR).select('==', True).new()


# Create an identity matrix
def identity(dim):
    I = new(dim, dim)
    for i in range(dim):
        I[i, i] = True
    return I


def resize(M: Matrix, nrows, ncols, inplace=False):
    if inplace:
        M.resize(nrows, ncols)
        return M
    M_new = M.dup()
    M_new.resize(nrows, ncols)
    return M_new


# Create a new matrix
def new(nrows, ncols=None, dtype=types.BOOL):
    if ncols is None:
        return Vector(dtype, nrows)
    return Matrix(dtype, nrows, ncols)


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

    # Add identy to the adjacency matrix
    R = addI(P1) if P2 is None else add(identity(dim), P2)
    if print_matrix:
        print('R = R2 + I = \n' + str(R) + '\n')

    # Iteratively compute the transitive closure using boolean matrix multiplication
    # Initialise closure matrix with an empty matrix
    R_ = new(dim, dim)
    while True:
        # R = R x R until no new connections are found
        R_ << square(R, inplace=True)
        if print_matrix:
            print('fixpoint = \n' + str(R_) + '\n')
        if R_.isequal(R):
            break
        R << R_

    # Multiply to remove redundant diagonal elements
    res = mul(R_, P1)

    if print_matrix:
        print('R0* = \n' + str(res) + '\n')

    return res


# GraphBLAS version of BMLP-SMP algorithm which performs vector multiplication
def SMP(V, R1, print_matrix=False):

    # Push the model subset selection into the summation for improved performance
    V_i = V.dup()
    V_ = V.dup()
    while True:
        # Apply vector multiplication (selection) to the transitive closure
        V_ << add(V_i, mul(V_i, R1, inplace=True), inplace=True)
        if print_matrix:
            print('V* = \n' + str(V_) + '\n')
        if V_.isequal(V_i):
            break
        V_i << V_

    # Multiply to remove redundant diagonal elements
    res = mul(V_, R1)

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
        R1 = load(R1)
        R2 = load(R2)
        nrows = R1.ncols
        ncols = max(R1.nrows, R2.ncols)
    # If R1 and R2 are not stored as SuiteSparse binary format
    # then R1 and R2 will be pre-processed as they have the same sizes
    # Otherwise, stored R1 has already been transposed and has a different size.
    else:
        if R1.ncols == R2.nrows:
            # R1 is transposed
            nrows = R1.ncols
            ncols = max(R1.nrows, R2.ncols)
        elif R1.nrows == R2.nrows:
            # R1 is not transposed
            nrows = R1.nrows
            ncols = max(R1.ncols, R2.ncols)
            R1 = resize(R1, nrows, ncols)
            R1 = transpose(R1)
            R2 = resize(R2, nrows, ncols)
        else:
            raise ValueError('Matrix dimensions do not match')

    # Need to resize matrices to be compatible due to leading zeros
    ninputs = V.nrows
    V_ = resize(V, ninputs, ncols)

    # When a filter on R2 is applied, negate it to find which R2 rows to keep
    if T is not None:
        T_ = resize(T, ninputs, nrows)
        T_ = negate(T_, inplace=True)

    SNum = 0
    V__ = new(ninputs, nrows)
    res = new(ninputs, ncols)
    V_neg = new(ninputs, ncols)
    M = new(ninputs, ncols)
    while True:
        # Find all rows that are subsets of V
        # Would need R2[i,j] -> R1[i,j] if matrix-matrix mul
        # can be performed on the element union

        # A sparse matrix workaround
        # Negate the V matrix while keeping it sparse to find all R1 rows that are not subsets
        # This can be done with sparse matrix mul without require union matrix-matrix mul
        V_neg << negate(V_, inplace=True)
        V__ << mul(V_neg, R1, inplace=True)
        V__ << negate(V__, inplace=True)

        # Multiply with rows in R2 filtered by T and update
        if T is None:
            M << gb.semiring.land_lor(V__ @ R2)
            res << gb.binary.lor(M | V_)
        else:
            V__ << gb.binary.land(V__ & T_)
            M << gb.semiring.land_lor(V__ @ R2)
            res << gb.binary.lor(M | V_)

        if print_matrix:
            print('V* = \n' + str(res) + '\n')
        if res.isequal(V_):
            break

        V_ << res
        res.clear()
        M.clear()
        V__.clear()
        SNum += 1

    if print_matrix:
        print('V* = \n' + str(res) + '\n')

    return res, SNum
