import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


GPU_DEVICE = torch.cuda.current_device()


# Save and load functions
def load(file_path):
    """Load sparse matrix from CSV file into PyTorch sparse tensor"""
    df = pd.read_csv(file_path)
    nrows = df['row'].iloc[0]
    ncols = df['col'].iloc[0]

    # Extract indices and values
    indices = torch.tensor([list(df['row'].iloc[1:].values), list(
        df['col'].iloc[1:].values)], dtype=torch.long)
    values = torch.ones(indices.shape[1], dtype=torch.bool)

    # Create tensor
    return torch.sparse_coo_tensor(indices, values, (nrows, ncols)).to_dense()


def save(matrix, file_path):
    """Save PyTorch sparse tensor to CSV file"""
    # Convert to COO format
    if not matrix.is_sparse:
        matrix = matrix.to_sparse()

    coo = matrix.to_sparse_coo().coalesce()
    indices = coo.indices()

    df = pd.DataFrame({
        'attribute': ['dim'] + ['elem'] * indices.shape[1],
        'row': np.insert(indices[0], 0, matrix.shape[0]),
        'col': np.insert(indices[1], 0, matrix.shape[1])
    })

    df.to_csv(file_path, index=False)


# Matrix operations
def mul(M1, M2):
    """Boolean matrix multiplication"""
    return M1 @ M2


def square(M):
    """Square a matrix using boolean operations"""
    return mul(M, M)


def add(M1, M2):
    """Add two matrices using boolean OR"""
    return (M1 + M2)


def intersection(M1, M2):
    """Find intersection using boolean AND"""
    return (M1 * M2)


def transpose(M):
    """Transpose a matrix"""
    return M.transpose(0, 1)


def addI(M):
    """Add identity matrix"""
    assert M.shape[0] == M.shape[1]
    return add(M, identity(M.shape[0]))


def negate(M):
    """Negate a boolean matrix"""
    # Create dense matrix of False values
    return ~M


def identity(dim):
    """Create identity matrix"""
    return torch.eye(dim, dtype=torch.bool, device=f'cuda:{GPU_DEVICE}')


def resize(M, nrows, ncols):
    """Resize matrix to new dimensions"""
    result = torch.zeros((nrows, ncols), dtype=M.dtype)
    min_rows = min(M.shape[0], nrows)
    min_cols = min(M.shape[1], ncols)
    result[:min_rows, :min_cols] = M[:min_rows, :min_cols]

    return result


# Create matrix
def new(nrows, ncols=None):
    """Create a new sparse matrix"""
    return torch.zeros(nrows, ncols)


def RMS(P1):
    """PyTorch version of BMLP-RMS algorithm"""

    # Add identity to the adjacency matrix
    R = addI(P1)

    # Iteratively compute the transitive closure
    R_ = square(R)
    while not torch.equal(R_, R):
        # R = R x R until no new connections are found
        R = R_
        R_ = square(R)

    # Multiply to remove redundant diagonal elements
    res = mul(R_, P1)

    return res


def RMS_query(P1, P2):
    """PyTorch version of BMLP-RMS algorithm"""
    # The dimensions of the matrix
    dim = P1.shape[0]

    # Add identity to the adjacency matrix
    R = add(identity(dim), P2)

    # Iteratively compute the transitive closure
    R_ = square(R)
    while not torch.equal(R_, R):
        # R = R x R until no new connections are found
        R = R_
        R_ = square(R)

    # Multiply to remove redundant diagonal elements
    res = mul(R_, P1)

    return res


def SMP(V, R1, print_matrix=False):
    """PyTorch version of BMLP-SMP algorithm"""
    # Push the model subset selection into summation
    V_i = V.clone()
    V_ = V.clone()

    while True:
        # Apply vector multiplication to the transitive closure
        V_ = add(V_i, mul(V_i, R1))
        if print_matrix:
            print('V* = \n' + str(V_.to_dense()) + '\n')
        if torch.equal(V_.to_dense(), V_i.to_dense()):
            break
        V_i = V_

    # Multiply to remove redundant diagonal elements
    res = mul(V_, R1)

    if print_matrix:
        print('V* = \n' + str(res.to_dense()) + '\n')

    return res


def IE(V, R1, R2, T=None, localised=False, print_matrix=False):
    """PyTorch version of BMLP-IE algorithm"""
    # Process input matrices
    if localised:
        R1 = load(R1)
        R2 = load(R2)
        nrows = R1.shape[1]
        ncols = max(R1.shape[0], R2.shape[1])
    else:
        if R1.shape[1] == R2.shape[0]:
            # R1 is transposed
            nrows = R1.shape[1]
            ncols = max(R1.shape[0], R2.shape[1])
        elif R1.shape[0] == R2.shape[0]:
            # R1 is not transposed
            nrows = R1.shape[0]
            ncols = max(R1.shape[1], R2.shape[1])
            R1 = resize(R1, nrows, ncols)
            R1 = transpose(R1)
            R2 = resize(R2, nrows, ncols)
        else:
            raise ValueError('Matrix dimensions do not match')

    # Resize matrices
    ninputs = V.shape[0]
    V_ = resize(V, ninputs, ncols)

    # Process filter
    if T is not None:
        T_ = resize(T, ninputs, nrows)
        T_ = negate(T_)

    SNum = 0
    V__ = new(ninputs, nrows)
    res = new(ninputs, ncols)
    V_neg = new(ninputs, ncols)
    M = new(ninputs, ncols)

    while True:
        # Find all rows that are subsets of V
        V_neg = negate(V_)
        V__ = mul(V_neg, R1)
        V__ = negate(V__)

        # Multiply with rows in R2 filtered by T and update
        if T is None:
            M = mul(V__, R2)
            res = add(M, V_)
        else:
            V__ = intersection(V__, T_)
            M = mul(V__, R2)
            res = add(M, V_)

        if print_matrix:
            print('V* = \n' + str(res.to_dense()) + '\n')

        if torch.equal(res.to_dense(), V_.to_dense()):
            break

        V_ = res
        SNum += 1

    if print_matrix:
        print('V* = \n' + str(res.to_dense()) + '\n')

    return res, SNum
