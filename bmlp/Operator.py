from abc import ABC, abstractmethod
from typing import Any
from .Predicate import *


class Operator(ABC):
    """Abstract base class for matrix operators"""

    @abstractmethod
    def apply(self, *args: Predicate, syms: Symbols) -> Predicate:
        """Apply the operator to the given predicates"""
        pass


class UnaryOp(Operator):
    """Abstract base class for matrix operators"""

    @abstractmethod
    def apply(self, pred: Predicate, syms: Symbols) -> Predicate:
        """Apply the operator to the given predicates"""
        pass


class BinaryOp(Operator):
    """Abstract base class for matrix operators"""

    @abstractmethod
    def apply(self, pred1: Predicate, pred2: Predicate, syms: Symbols) -> Predicate:
        """Apply the operator to the given predicates"""
        pass


# Inverse a predicate object using matrix transpose
class InvOp(UnaryOp):
    def apply(self, pred: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(pred.get_matrix().T, new_sym,
                         expr=f'{pred}{new_sym}(X, Y) :- {pred.get_name()}(Y, X).\n')


# Create the negation of M using sparse matrix
class NegOp(UnaryOp):
    def apply(self, pred: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(matrix_negate(pred.get_matrix()), new_sym,
                         expr=f'{pred}{new_sym}(X, Y) :- not{pred.get_name()}(X, Y).\n')


# Create the conjunction of two predicate objects if they share the same variables
class ConjOp(BinaryOp):
    def apply(self, pred1: Predicate, pred2: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(pred1.get_matrix() * pred2.get_matrix(), new_sym,
                         expr=f'{pred1}{pred2}{new_sym}(X, Y) :- {pred1.get_name()}(X, Y), {pred2.get_name()}(X, Y).\n')


# Create the disjunction of two predicate objects if they share the same variables
class DisjOp(BinaryOp):
    def apply(self, pred1: Predicate, pred2: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(pred1.get_matrix() + pred2.get_matrix(), new_sym,
                         expr=f"{pred1}{pred2}{new_sym}(X, Y) :- {pred1.get_name()}(X, Y).\n{new_sym}(X, Y) :- {pred2.get_name()}(X, Y).\n")


# Create a linear recursive predicate object from base case M1 and body M2
class RecursOp1(BinaryOp):
    def apply(self, pred1: Predicate, pred2: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(BMLP_RMS(pred1.get_matrix(), pred2.get_matrix()), new_sym,
                         expr=f"{pred1}{pred2}{new_sym}(X, Y) :- {pred1.get_name()}(X, Y).\n{new_sym}(X, Y) :- {pred2.get_name()}(X, Z), {new_sym}(Z, Y).\n")


class RecursOp2(BinaryOp):
    def apply(self, pred1: Predicate, pred2: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(
            BMLP_RMS(pred2.get_matrix(), pred1.get_matrix()), new_sym,
            expr=f"{pred1}{pred2}{new_sym}(X, Y) :- {pred2.get_name()}(X, Y).\n{new_sym}(X, Y) :- {pred1.get_name()}(X, Z), {new_sym}(Z, Y).\n",
        )


class RecursOpSelf(UnaryOp):
    def apply(self, pred: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(
            BMLP_RMS(pred.get_matrix(), pred.get_matrix()), new_sym,
            expr=f"{pred}{new_sym}(X, Y) :- {pred.get_name()}(X, Y).\n{new_sym}(X, Y) :- {pred.get_name()}(X, Z), {new_sym}(Z, Y).\n",
        )


# Create a chain of two predicate objects
class ChainOp1(BinaryOp):
    def apply(self, pred1: Predicate, pred2: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(pred1.get_matrix() @ pred2.get_matrix(), new_sym,
                         expr=f"{pred1}{pred2}{new_sym}(X, Y) :- {pred1.get_name()}(X, Z), {pred2.get_name()}(Z, Y).\n")


class ChainOp2(BinaryOp):
    def apply(self, pred1: Predicate, pred2: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(pred2.get_matrix() @ pred1.get_matrix(), new_sym,
                         expr=f"{pred1}{pred2}{new_sym}(X, Y) :- {pred2.get_name()}(X, Z), {pred1.get_name()}(Z, Y).\n")


class ChainOpSelf(UnaryOp):
    def apply(self, pred: Predicate, syms: Symbols) -> Predicate:
        new_sym = syms.next_symbol()
        return Predicate(pred.get_matrix() @ pred.get_matrix(), new_sym,
                         expr=f"{pred}{new_sym}(X, Y) :- {pred.get_name()}(X, Z), {pred.get_name()}(Z, Y).\n")
