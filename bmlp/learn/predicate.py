from ..core.matrix import *
from ..core.utils import *
from cudf import DataFrame


class Predicate:
    def __init__(self, matrix: Matrix, name: str, pos_score: float = np.inf, neg_score: float = -np.inf, expr: str = ''):
        """
        Initialize a Predicate object

        Args:
            matrix: The predicate matrix
            name: Name of the predicate
            pos_score: Positive score (default np.inf)
            neg_score: Negative score (default -np.inf)
            expr: Program expression (default "")
        """
        self.matrix = matrix
        self.name = name
        self.pos_score = pos_score
        self.neg_score = neg_score
        self.expr = expr

    def get_expr(self):
        """Return the program in string"""
        return self.expr

    def set_expr(self, expr):
        """Set program string"""
        self.expr = expr

    def get_matrix(self):
        """Return predicate matrix"""
        return self.matrix

    def get_name(self):
        """Return the predicate in string format"""
        return self.name

    def get_scores(self):
        """Return tuple of (pos_score, neg_score)"""
        return (self.pos_score, self.neg_score)

    def set_scores(self, pos_score, neg_score):
        """Set predicate scores"""
        self.pos_score = pos_score
        self.neg_score = neg_score

    def get_positive_score(self):
        """Return the positive score"""
        return self.pos_score

    def get_negative_score(self):
        """Return the negative score"""
        return self.neg_score

    def set_positive_score(self, pos_score):
        """Set the positive score"""
        self.pos_score = pos_score

    def set_negative_score(self, neg_score):
        """Set the negative score"""
        self.neg_score = neg_score

    def update_matrix(self, new_matrix):
        """Update predicate matrix"""
        self.matrix = new_matrix

    def __str__(self):
        """String representation"""
        return self.expr

    def __repr__(self):
        """Detailed representation"""
        return f"Predicate(\nmatrix=\n{self.matrix}, \nname={self.name}, scores=({self.pos_score}, {self.neg_score}))"

    def hash(self) -> int:
        """Calculate the hash value for the instance.

        This method generates a hash value by converting the matrix to a list and summing the hash values of its elements. 
        https://docs.rapids.ai/api/cudf/nightly/user_guide/api_docs/api/cudf.dataframe.hash_values/

        Returns:
            int: The computed hash value of the matrix.
        """
        coo = self.matrix.to_coo(values=False)
        return int(DataFrame(coo[:-1]).hash_values().sum())

    # def __hash__(self):
    #     """
    #     Mueller Hash function for sparse matrices based on WarpCore.
    #     https://github.com/sleeepyjack/warpcore/blob/master/include/warpcore/hashers.cuh#L50

    #     Assuming matrices have dimensions multiples of 64.
    #     """
    #     # Pseudo code
    #     for serialise matrix data x:
    #         x = ((x >> 16) ^ x) * 0x45d9f3b
    #         x = ((x >> 16) ^ x) * 0x45d9f3b
    #         x = ((x >> 16) ^ x)
    #
    #     return x

    # def __hash__(self):
    #     """
    #     Simple non-cryptographic XOR Hash function of sparse matrices.
    #     Avoid some solutions actually.
    #     """

    #     # Apply XOR Monoid reduction to the matrix
    #     bit_position = self.matrix.reduce_vector(
    #         types.BOOL.LXOR_MONOID).to_arrays()
    #     matrix_hash = 0

    #     # Convert the bit position to a hash
    #     for i, pos in enumerate(bit_position[0]):
    #         matrix_hash |= 1 << pos if bit_position[1][i] == 1 else 0
    #     print(bit_position)
    #     print(matrix_hash)
    #     # Convert the bit position to a hash
    #     # for i in bit_position:
    #     #     matrix_hash |= 1 << i

    #     # if matrix_hash == 31:
    #     #     print(self.matrix.reduce_vector(
    #     #         types.BOOL.LXOR_MONOID).to_arrays())
    #     #     print(self.matrix.reduce_vector(
    #     #         types.BOOL.LXOR_MONOID).to_arrays()[1][4])
    #     return matrix_hash


class Symbols:
    def __init__(self, first_symbol, symbol_base='inv'):
        self.current = first_symbol
        self.symbol_base = symbol_base

    def next_symbol(self):
        symbol = self.current
        self.current += 1
        return f'{self.symbol_base}_{symbol}'


# Initial a predicate object with a matrix and name string
def new_predicate(M, name):
    return Predicate(M, name)
