from .Matrix import *
from .Utils import *


from .Matrix import *
from .Utils import *


class Predicate:
    def __init__(self, matrix: Matrix.sparse, name, pos_score=0.0, neg_score=0.0):
        """
        Initialize a Predicate object

        Args:
            matrix: The predicate matrix
            name: Name of the predicate
            pos_score: Positive score (default 0.0)
            neg_score: Negative score (default 0.0)
        """
        self.matrix = matrix
        self.name = name
        self.pos_score = pos_score
        self.neg_score = neg_score

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
        return f"Predicate({self.name})"

    def __repr__(self):
        """Detailed representation"""
        return f"Predicate(\nmatrix=\n{self.matrix}, \nname={self.name}, scores=({self.pos_score}, {self.neg_score}))"

    def __hash__(self):
        """
        Simple non-cryptographic XOR Hash function of sparse matrices.
        """

        # Apply XOR Monoid reduction to the matrix
        bit_position = self.matrix.reduce_vector(
            types.BOOL.LXOR_MONOID).to_arrays()[0]
        matrix_hash = 0

        # Convert the bit position to a hash
        for i in bit_position:
            matrix_hash |= 1 << i

        return matrix_hash


# Initial a predicate object with a matrix and name string
def new_predicate(M, name):
    return Predicate(M, name)
