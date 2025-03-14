from graphblas import Matrix
from graphblas.dtypes import *
import torch


# Parse a boolean matrix from Prolog version of BMLP which has integers as rows
# return a list of bitcodes arrays, the number of facts and the maximum length of the bitcodes
def parse_prolog_binary_codes(path):

    codes = []
    max_length = 0

    with open(path, 'r') as prolog:

        num_facts = 0

        for row in prolog:
            if "%" not in row and row != "\n":

                # Get the integer representation of the bitcode
                code, length = integer_to_binary_code(
                    int(row.replace(" ", "").replace("\n", "").split(",")[1].strip(").")))
                codes.append(code)
                max_length = max(max_length, length)
                num_facts += 1

    return codes, num_facts, max_length,


def integer_to_binary_code(n):
    len = n.bit_length()
    return [n >> i & 1 for i in range(0, len)], len


def boolean_matrix_to_integers(matrix: Matrix, name='abdm', path='cm.pl'):

    cm = []
    is_using_tensor = torch.cuda.is_available() and isinstance(matrix, torch.Tensor)

    if is_using_tensor:
        nrows = matrix.shape[0]
        ncols = matrix.shape[1]
    else:
        nrows = matrix.nrows
        ncols = matrix.ncols

    with open(path, 'w') as prolog:
        for i in range(nrows):

            out = 0
            for j in range(ncols - 1, -1, -1):

                # sparse matrix elements can be empty bits
                if is_using_tensor:
                    element = matrix[i, j].item()
                    out = (out << 1) | (element > 0.0)
                else:
                    element = matrix.get(i, j)
                    # element may be None
                    out = (out << 1) | element if element else (out << 1)

            prolog.write('%s(%s,%s).\n' % (name, i, out))
            cm.append(out)
    return cm


# From a path to a prolog file containing a boolean matrix
# convert it into a graphBLAS matrix for computation
def integers_to_boolean_matrix(path, is_squared=False, to_tensor=False):

    is_using_tensor = torch.cuda.is_available() and to_tensor
    bitcodes, nrows, ncols = parse_prolog_binary_codes(path)

    # If the matrix needs to be squared, create a square matrix
    if is_squared:
        dim = max(nrows, ncols)
        if is_using_tensor:
            matrix = torch.zeros(dim, dim)
        else:
            matrix = Matrix(BOOL, dim, dim)
    else:
        if is_using_tensor:
            matrix = torch.zeros(nrows, ncols)
        else:
            matrix = Matrix(BOOL, nrows, ncols)

    for row in range(len(bitcodes)):
        for col in range(len(bitcodes[row])):

            if bitcodes[row][col]:
                if is_using_tensor:
                    matrix[row, col] = 1.0
                else:
                    matrix[row, col] = True

    return matrix


# From a python list of lists to a graphBLAS matrix
def lists_to_matrix(lists: list[list[int]]):
    matrix = Matrix.from_dense(lists, missing_value=0)
    return matrix
