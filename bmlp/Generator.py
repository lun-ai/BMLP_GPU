from .Operator import *
from .Task import *


# unary_ops = [pinv, precur_self]
# With Negation
# unary_ops = [pinv, pneg, precur_self]
# binary_ops = [pconj, pdisj, precur_1, precur_2, pchain_1, pchain_2]


# Generate more predicates from existing predicates using unary and binary operators
class Generator:
    def __init__(self: Any,
                 predicates: list = [],
                 unary_ops: list[UnaryOp] = [
                     InvOp, RecursOpSelf, ChainOpSelf],
                 binary_ops: list[BinaryOp] = [ConjOp, DisjOp, RecursOp1, RecursOp2, ChainOp1, ChainOp1]):
        self.predicates = predicates
        self.unary_ops = unary_ops  # Based on allowed unary operators
        self.binary_ops = binary_ops  # Based on allowed binary operators
        self.redundancy = 0
        self.new_preds = []
        # Use the default symbol creator
        self.syms = Symbols(1)

    def add_predicates(self, predicates):
        """Add predicates for generation"""
        self.predicates.append(predicates)

    def update_predicates(self, predicates):
        """Clear all stored predicates"""
        self.predicates = predicates

    def clear_predicates(self):
        """Clear all stored predicates"""
        self.predicates = []

    def get_predicates(self):
        """Return current predicates"""
        return self.predicates

    # Apply operators to a set of predicate object
    def invent_predicates_for_task(self, target: Task, cache: dict = {}, num_hypo: int = 1):

        if target is None:
            raise ValueError(
                "Invalid Task: required positive, negative examples and optimality to be defined!")

        # New predicates objects generated
        self.new_preds = []
        self.redundancy = 0

        # Apply unary operators
        for pred in self.predicates:
            for unary_op in self.unary_ops:
                new_pred = unary_op.apply(unary_op, pred, self.syms)
                if hash(new_pred) not in cache:
                    cache[hash(new_pred)] = True
                    self.new_preds.append(new_pred)
                else:
                    self.redundancy += 1

        # Apply binary operators to every matrix and the rest
        for i in range(len(self.predicates) - 1):
            for j in range(i + 1, len(self.predicates)):
                for binary_op in self.binary_ops:
                    fst_pred = self.predicates[i]
                    snd_pred = self.predicates[j]
                    new_pred = binary_op.apply(
                        binary_op, fst_pred, snd_pred, self.syms)
                    if hash(new_pred) not in cache:
                        cache[hash(new_pred)] = True
                        self.new_preds.append(new_pred)
                    else:
                        self.redundancy += 1

        # Check if any of the new predicates are correct
        correct_preds = []
        for pred in self.new_preds:
            if target.check_correctness(pred):
                correct_preds.append(pred)
                if len(correct_preds) >= num_hypo:
                    break

        return len(correct_preds) >= num_hypo, correct_preds, self.new_preds

    def get_redundancy(self):
        """Retrieve the redundancy value.

        This method returns the redundancy of all generated matrices from the cache. 

        Returns:
            float: The redundancy count.
        """
        return self.redundancy

    def get_new_preds(self):
        """Retrieve the latest predictions.

        This method returns the `new_preds` attribute, which holds the most recent predicates generated 
        by the instance. 

        Returns:
            list: The new predicates in the latest generation.
        """
        return self.new_preds
