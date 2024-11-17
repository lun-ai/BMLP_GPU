from typing import Any
from .Predicate import *


# Inverse a predicate object using matrix transpose
def pinv(pred):
    return new_predicate(pred.get_matrix().T, f"({pred.get_name()}.T)")


# Create the negation of M using sparse matrix
def pneg(pred):
    return new_predicate(matrix_negate(pred.get_matrix()), f"(~ {pred.get_name()})")


# Create the conjunction of two predicate objects if they share the same variables
def pconj(pred1, pred2):
    return new_predicate(pred1.get_matrix() * pred2.get_matrix(), f"({pred1.get_name()}, {pred2.get_name()})")


# Create the disjunction of two predicate objects if they share the same variables
def pdisj(pred1, pred2):
    return new_predicate(pred1.get_matrix() + pred2.get_matrix(), f"({pred1.get_name()} ¦ {pred2.get_name()})")


# Create a linear recursive predicate object from base case M1 and body M2
def precur_1(pred1, pred2):
    return new_predicate(
        BMLP_RMS(pred1.get_matrix(), pred2.get_matrix()),
        f"(inv <- {pred1.get_name()} ¦ inv <- {pred2.get_name()}, inv)",
    )


def precur_2(pred1, pred2):
    return new_predicate(
        BMLP_RMS(pred2.get_matrix(), pred1.get_matrix()),
        f"(inv <- {pred2.get_name()} ¦ inv <- {pred1.get_name()}, inv)",
    )


def precur_self(pred):
    return new_predicate(
        BMLP_RMS(pred.get_matrix(), pred.get_matrix()),
        f"(inv <- {pred.get_name()} ¦ inv <- {pred.get_name()}, inv)",
    )


# Create a chain of two predicate objects
def pchain_1(pred1, pred2):
    return new_predicate(pred1.get_matrix() @ pred2.get_matrix(), f"(inv <- {pred1.get_name()}, {pred2.get_name()})")


def pchain_2(pred1, pred2):
    return new_predicate(pred2.get_matrix() @ pred1.get_matrix(), f"(inv <- {pred2.get_name()}, {pred1.get_name()})")


def pchain_self(pred):
    return new_predicate(pred.get_matrix() @ pred.get_matrix(), f"(inv <- {pred.get_name()}, {pred.get_name()})")
