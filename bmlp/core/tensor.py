import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Save and load functions
def load(file_path):
    """Load sparse matrix from CSV file into PyTorch sparse tensor"""
    df = pd.read_csv(file_path)
    nrows = df['row'].iloc[0]
    ncols = df['col'].iloc[0]
    
    # Extract indices and values
    indices = torch.tensor([list(df['row'].iloc[1:].values), list(df['col'].iloc[1:].values)], dtype=torch.long)
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
    result = torch.sparse.mm(M1, M2).bool()
    return result

def square(M):
    """Square a matrix using boolean operations"""
    return mul(M, M)

def add(M1, M2):
    """Add two matrices using boolean OR"""
    result = (M1 | M2).bool()
    return result

def intersection(M1, M2):
    """Find intersection using boolean AND"""
    if M1.is_sparse and M2.is_sparse:
        # Convert to dense for boolean AND, then back to sparse
        result = (M1.to_dense() & M2.to_dense()).to_sparse()
    else:
        result = (M1.to_dense() & M2.to_dense()).to_sparse()
    return result

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
    if M.is_sparse:
        dense_M = M.to_dense()
    else:
        dense_M = M
    result = (~dense_M).to_sparse()
    return result

def identity(dim):
    """Create identity matrix"""
    return torch.eye(dim, dtype=torch.bool).to_sparse()

def resize(M, nrows, ncols, inplace=False):
    """Resize matrix to new dimensions"""
    if M.is_sparse:
        indices = M.indices()
        values = M.values()
        
        # Filter indices that are within the new dimensions
        mask = (indices[0] < nrows) & (indices[1] < ncols)
        new_indices = indices[:, mask]
        new_values = values[mask]
        
        result = torch.sparse_coo_tensor(new_indices, new_values, (nrows, ncols))
    else:
        result = torch.zeros((nrows, ncols), dtype=M.dtype)
        min_rows = min(M.shape[0], nrows)
        min_cols = min(M.shape[1], ncols)
        result[:min_rows, :min_cols] = M[:min_rows, :min_cols]
    
    return result

def new(nrows, ncols=None, dtype=torch.bool):
    """Create a new sparse matrix"""
    if ncols is None:
        # Create vector
        return torch.sparse_coo_tensor(torch.zeros((1, 0), dtype=torch.long), 
                                      torch.zeros(0, dtype=dtype),
                                      (nrows,))
    # Create matrix
    return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), 
                                  torch.zeros(0, dtype=dtype),
                                  (nrows, ncols))

def RMS(P1, P2=None, print_matrix=False):
    """PyTorch version of BMLP-RMS algorithm"""
    # The dimensions of the matrix
    dim = P1.shape[0]
    
    # Add identity to the adjacency matrix
    R = addI(P1) if P2 is None else add(identity(dim), P2)
    if print_matrix:
        print('R = R2 + I = \n' + str(R.to_dense()) + '\n')
    
    # Iteratively compute the transitive closure
    R_ = new(dim, dim)
    while True:
        # R = R x R until no new connections are found
        R_ = square(R)
        if print_matrix:
            print('fixpoint = \n' + str(R_.to_dense()) + '\n')
        if torch.equal(R_.to_dense(), R.to_dense()):
            break
        R = R_
    
    # Multiply to remove redundant diagonal elements
    res = mul(R_, P1)
    
    if print_matrix:
        print('R0* = \n' + str(res.to_dense()) + '\n')
    
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