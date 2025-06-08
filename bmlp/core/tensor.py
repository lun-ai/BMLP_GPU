import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float16)

GPU_DEVICE = f'cuda:{torch.cuda.current_device()}'
# Use float16 for faster computation on GPU
# Values above 0.0 are considered True, 0.0 is False
D_TYPE = torch.float16


# Save and load functions
def load(file_path):
    """Load sparse matrix from CSV file into PyTorch sparse tensor"""
    df = pd.read_csv(file_path)
    nrows = df['row'].iloc[0]
    ncols = df['col'].iloc[0]

    # Extract indices and values
    indices = torch.tensor([list(df['row'].iloc[1:].values), list(
        df['col'].iloc[1:].values)])
    values = torch.ones(indices.shape[1])

    # Create tensor
    return torch.sparse_coo_tensor(indices, values, (nrows, ncols), dtype=D_TYPE).to_dense()


def save(matrix, file_path):
    """Save PyTorch tensor to CSV file"""
    # Need to coalesce to avoid duplicate indices and copy to CPU memory first
    coo = matrix.to_sparse_coo().coalesce().cpu()
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
    return torch.clamp(M1 @ M2, min=0.0, max=1.0)


def square(M):
    """Square a matrix using boolean operations"""
    return mul(M, M)


def add(M1, M2):
    """Add two matrices using boolean OR"""
    return torch.clamp(M1 + M2, min=0.0, max=1.0)


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
    return torch.ones(M.shape) - torch.clamp(M, max=1.0)


def identity(dim):
    """Create identity matrix"""
    return torch.eye(dim)


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
    if ncols is None:
        return torch.zeros(nrows, dtype=D_TYPE)
    return torch.zeros(nrows, ncols, dtype=D_TYPE)


def RMS(P1, P2=None):
    """PyTorch version of BMLP-RMS algorithm"""

    # Add identity to the adjacency matrix
    dim = P1.shape[0]
    if P2 is None:
        R = addI(P1)
    else:
        R = add(identity(dim), P2)

    # Iteratively compute the transitive closure
    R_ = square(R)
    while not torch.equal(R_, R):
        R = R_
        # R = R x R until no new connections are found
        R_ = square(R)

    # Multiply to remove redundant diagonal elements
    res = mul(R_, P1)

    return res


def SMP(V, R1):
    """PyTorch version of BMLP-SMP algorithm"""
    # Push the model subset selection into summation
    V_i = V
    V_ = add(V_i, mul(V_i, R1))

    while not torch.equal(V_, V_i):
        V_i = V_
        # Apply vector multiplication to the transitive closure
        V_ = add(V_i, mul(V_i, R1))

    # Multiply to remove redundant diagonal elements
    res = mul(V_, R1)

    return res


def IE(V, R1, R2, T=None, localised=False):
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

    # Resize matrices (avoid dimension mismatch from sparse tensor)
    ninputs = V.shape[0]
    V_ = resize(V, ninputs, ncols)

    # Process filter
    if T is not None:
        T_ = negate(resize(T, ninputs, nrows))

    SNum = 0
    V__ = new(ninputs, nrows)
    res = new(ninputs, ncols)

    while True:
        # Find all rows that are subsets of V
        V__ = mul(negate(V_), R1)
        V__ = negate(V__)

        # Multiply with rows in R2 filtered by T and update
        if T is None:
            res = add(mul(V__, R2), V_)
        else:
            V__ = intersection(V__, T_)
            res = add(mul(V__, R2), V_)

        # Check if the result has changed
        if torch.equal(res, V_):
            break

        V_ = res
        SNum += 1

    return res, SNum
