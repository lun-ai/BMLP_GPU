import pygraphblas as gb
import numpy as np
from bmlp.utils import *

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


# Write graphBLAS matrix as a set of integers in a prolog file
def boolean_matrix_to_integers(matrix, name, path):
    
    with open(path, 'w') as prolog:
        for i in range(0, matrix.nrows):
            
            out = 0
            for j in range(matrix.ncols - 1, -1, -1):
                
                # sparse matrix elements can be empty bits
                element = matrix.get(i, j)
                out = (out << 1) | element if element else (out << 1)
                
            prolog.write('%s(%s,%s).\n' % (name, i, out))


# From a path to a prolog file containing a boolean matrix
# convert it into a graphBLAS matrix for computation
def integers_to_boolean_matrix(path):
    
    bitcodes, dim = parse_prolog_binary_codes(path)
    matrix = gb.Matrix.sparse(gb.BOOL, dim, dim)
    
    for row in range(0, len(bitcodes)):
        for col in range(0, len(bitcodes[row])):
            
            if bitcodes[row][col]:
                matrix[row, col] = True
    
    return matrix
    

# Create an identity matrix
def identity(dim):
    I = gb.Matrix.sparse(gb.BOOL, dim, dim)
    for i in range(dim):
        I[i, i] = True
    return I



def BMLP_RMS(P1, P2=None, print_matrix=False):
    
    """GraphBLAS version of BMLP-RMS algorithm which performs repeated matrix squaring

        P0, P1 and P2 are boolean matrices representing the following predicates.
        p0(X,Z):- p1(X,Z). 
        p0(X,Z):- p2(X,Y),p0(Y,Z).

        Default p1 and p2 represent the same predicate.

    Args:
        P1 (Matrix.sparse): boolean matrix for the non recursive body p1 or default both p1 and p2
        P2 (Matrix.sparse, optional): boolean matrix for the recursive body p2 if differs from p1. Defaults to None.
        print_matrix (bool, optional): print trace of fixpoint computation. Defaults to False.

    Returns:
        Matrix.sparse: fixpoint boolean matrix representing predicate p0
    """
    
    # the dimensions of the matrix, e.g. the number of nodes in a graph
    dim = P1.nrows
    empty_matrix = gb.Matrix.sparse(gb.BOOL, dim, dim)

    # Add identy to the adjacency matrix
    R = identity(dim) + P1 if P2 is None else identity(dim) + P2
    if print_matrix:
        print('R = R2 + I = \n'+ str(R) + '\n')

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
def BMLP_SMP(V, R1, print_matrix=False):
    
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


def BMLP_IE(V, R1, R2, T=None, print_matrix=False):
    """GraphBLAS version of BMLP-SMP algorithm 
        which performs matrix operations using two input matrices
        of dimension k x n

    Args:
        V (Vector.sparse): a 1 x 
        R1 (Matrix.sparse): _description_
        R2 (Matrix.sparse): _description_
        T (Vector.sparse, optional): _description_. Defaults to None.
        print_matrix (bool, optional): _description_. Defaults to False.

    Returns:
        Matrix.sparse: _description_
    """
    nrows = R1.nrows
    res = V
    V_ = gb.Vector.sparse(gb.BOOL, nrows)
    
    while True:
        # Find all rows that are subsets of V 
        for i in range(nrows):
            row = R1[i,:]
            if row.iseq(V * row):
                V_[i] = True

        # Multiply with rows in R2 filtered by T and update
        res = V_ @ R2 + V if T is None else (V_ * T) @ R2 + V
        if print_matrix:
            print('V* = \n' + str(res) + '\n')
        if res.iseq(V):
            break
        V = res
    
    if print_matrix:
        print('V* = \n' + str(res) + '\n')
    
    return res