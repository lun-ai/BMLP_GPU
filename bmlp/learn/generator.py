from .operator import *
from .task import *
import time


DEFAULT_UNARY_OPS = [InvOp, RecursOpSelf, ChainOpSelf]
DEFAULT_BINARY_OPS = [ConjOp, DisjOp, RecursOp1, RecursOp2, ChainOp1, ChainOp1]


# Generate more predicates from existing predicates using unary and binary operators
class Generator:
    def __init__(self: Any,
                 predicates: list = [],
                 unary_ops: list[UnaryOp] = DEFAULT_UNARY_OPS,
                 binary_ops: list[BinaryOp] = DEFAULT_BINARY_OPS):
        self.predicates = predicates
        self.unary_ops = unary_ops  # Based on allowed unary operators
        self.binary_ops = binary_ops  # Based on allowed binary operators
        self.redundancy = 0
        self.elimination = 0
        self.num_new_predicate = len(self.predicates)
        self.new_preds = []
        # Use the default symbol creator
        self.syms = Symbols(1)

    def add_predicates(self, predicates):
        """Add predicates for generation"""
        self.predicates.append(predicates)

    def update_predicates(self, num_new_predicate, predicates):
        """Clear all stored predicates"""
        self.predicates = predicates
        self.num_new_predicate = num_new_predicate

    def clear_predicates(self):
        """Clear all stored predicates"""
        self.predicates = []

    def get_predicates(self):
        """Return current predicates"""
        return self.predicates

    # Apply operators to a set of predicate object
    def invent_predicates_for_task(self, target: Task, cached: dict = {}, num_hypo: int = 1):

        if target is None:
            raise ValueError(
                "Invalid Task: required positive, negative examples and optimality to be defined!")

        # New predicates objects generated
        self.new_preds = []
        self.redundancy = 0
        self.elimination = 0

        # Apply unary operators
        for pred in self.predicates:
            for unary_op in self.unary_ops:
                new_pred = unary_op.apply(unary_op, pred, self.syms)
                if new_pred is None:
                    self.elimination += 1
                else:
                    # If input predicates are neither incomplete nor inconsistent
                    hash = new_pred.hash()
                    if hash not in cached:
                        cached[hash] = True
                        self.new_preds.append(new_pred)
                    else:
                        self.redundancy += 1

        # Apply binary operators to every matrix and the rest
        for i in range(self.num_new_predicate - 1):
            for j in range(i + 1, len(self.predicates)):
                for binary_op in self.binary_ops:
                    fst_pred = self.predicates[i]
                    snd_pred = self.predicates[j]
                    new_pred = binary_op.apply(
                        binary_op, fst_pred, snd_pred, self.syms)
                    # If input predicates are neither incomplete nor inconsistent
                    if new_pred is None:
                        self.elimination += 1
                    else:
                        hash = new_pred.hash()
                        if hash not in cached:
                            cached[hash] = True
                            self.new_preds.append(new_pred)
                        else:
                            self.redundancy += 1

        # Check if any of the new predicates are correct
        correct_preds = []
        for pred in self.new_preds:
            # Check if any of the new predicates are correct (consistent and complete)
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

    def get_elimination(self):
        """Retrieve the elimination value.

        This method returns the eliminated matrices from the hypothesis space. 

        Returns:
            float: The elimination count.
        """
        return self.elimination

    def get_new_preds(self):
        """Retrieve the latest predictions.

        This method returns the `new_preds` attribute, which holds the most recent predicates generated 
        by the instance. 

        Returns:
            list: The new predicates in the latest generation.
        """
        return self.new_preds
