
from abc import ABC, abstractmethod
from typing import Any, Optional
from .predicate import *
from .operator import UnaryOp, BinaryOp


class Expression(ABC):
    """Abstract base class for expressions in the syntax tree"""

    @abstractmethod
    def get_string(self) -> str:
        """Convert expression to string representation"""
        pass


class PredExpr(Expression):
    """Leaf node containing a predicate"""

    def __init__(self, predicate: Predicate):
        self.predicate = predicate

    def get_string(self) -> str:
        return str(self.predicate.get_expr())


class UnaryExpr(Expression):
    """Expression with one child and a unary operator"""

    def __init__(self, operator: UnaryOp, operand: Predicate):
        self.operator = operator
        self.operand = operand

    def get_string(self, head_pred: Predicate) -> str:
        return self.operator.get_string(head_pred, self.operand)


class BinaryExpr(Expression):
    """Expression with two children and a binary operator"""

    def __init__(self, operator: BinaryOp, left: Predicate, right: Predicate):
        self.operator = operator
        self.left = left
        self.right = right

    def get_string(self) -> str:
        return f"({str(self.left)} {self.operator.value} {str(self.right)})"


def create_expression(pred: Predicate):
    return PredExpr(pred)


def create_expression(op: UnaryExpr, pred: Predicate):
    return UnaryExpr(op, pred)


def create_expression(op: BinaryExpr, pred1: Predicate, pred2: Predicate):
    return BinaryExpr(op, pred1, pred2)
