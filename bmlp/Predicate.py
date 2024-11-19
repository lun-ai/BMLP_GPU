from .Matrix import *
from .Utils import *


class Predicate:
    def __init__(self, matrix: Matrix.sparse, name: str, pos_score: float = 0.0, neg_score: float = 0.0, expr: str = ''):
        """
        Initialize a Predicate object

        Args:
            matrix: The predicate matrix
            name: Name of the predicate
            pos_score: Positive score (default 0.0)
            neg_score: Negative score (default 0.0)
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

    def update_matrix(self, new_matrix):
        """Update predicate matrix"""
        self.matrix = new_matrix

    def __str__(self):
        """String representation"""
        return self.expr

    def __repr__(self):
        """Detailed representation"""
        return f"Predicate(\nmatrix=\n{self.matrix}, \nname={self.name}, scores=({self.pos_score}, {self.neg_score}))"

    # def __hash__(self):
    #     """
    #     Mueller Hash function for sparse matrices based on WarpCore.
    #     https://github.com/sleeepyjack/warpcore/blob/master/include/warpcore/hashers.cuh#L50

    #     Assuming matrices have dimensions multiples of 64.
    #     """

    #     matrix_hash = 0

    #     # Apply XOR Monoid reduction to the matrix
    #     for i in self.matrix.rows:
    #         x = ((x >> 16) ^ x) * 0x45d9f3b
    #     x = ((x >> 16) ^ x) * 0x45d9f3b
    #     x = ((x >> 16) ^ x)
    #     # Convert the bit position to a hash
    #     for i in bit_position:
    #         matrix_hash |= 1 << i

    #     return matrix_hash

    def __hash__(self):
        """
        Simple non-cryptographic XOR Hash function of sparse matrices.
        Avoid some solutions actually.
        """

        # Apply XOR Monoid reduction to the matrix
        bit_position = self.matrix.reduce_vector(
            types.BOOL.LXOR_MONOID).to_arrays()[0]
        matrix_hash = 0

        # Convert the bit position to a hash
        for i in bit_position:
            matrix_hash |= 1 << i

        return matrix_hash


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
