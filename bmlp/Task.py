from .Predicate import *


class Task:
    def __init__(self, pos: Matrix.sparse, neg: Matrix.sparse):
        """
        Initialize Task with positive and negative examples

        Args:
            Pos: Positive examples
            Neg: Negative examples
        """
        self.pos = pos
        self.neg = neg
        # Number of positive and negative examples
        self.PN = pos.reduce_int()
        self.NN = neg.reduce_int()

    # def validate(self):
    #     """Validate that all parameters are valid matrices"""
    #     return all(isinstance(x, (list, np.ndarray))
    #                for x in [self.pos, self.neg])

    def get_pos(self):
        """Return positive examples"""
        return self.pos

    def get_neg(self):
        """Return negative examples"""
        return self.neg

    def get_pn(self):
        """Return No. positive examples"""
        return self.PN

    def get_nn(self):
        """Return No. negative examples"""
        return self.NN

    def set_positive(self, new_pos):
        """Set new positive examples"""
        self.pos = new_pos

    def set_negative(self, new_neg):
        """Set new negative examples"""
        self.neg = new_neg

    def merge_tasks(self, other_task):
        """Merge another task with this one"""
        self.pos.extend(other_task.get_pos())
        self.neg.extend(other_task.get_neg())
        self.PN.extend(other_task.get_pn())
        self.NN.extend(other_task.get_nn())

    # Compute the coverage of invented predicates of positive and negative examples
    def pos_coverage(self, pred):
        return (pred.get_matrix() * self.pos).reduce_int() / self.PN if self.PN != 0 else 0

    def neg_coverage(self, pred):
        return (pred.get_matrix() * self.neg).reduce_int() / self.NN if self.NN != 0 else 0

    def check_correctness(self, pred):
        """Determine if the prediction is correct based on coverage metrics.

        This method evaluates the correctness of a prediction by checking if the positive coverage is 
        at least 1.0 and the negative coverage is at most 0.0. It checks if a predicate entails all 
        positive examples and none of the negative examples.

        Args:
            Pred (any): The prediction to be evaluated.

        Returns:
            bool: True if the prediction is correct based on the coverage criteria, otherwise False.
        """

        pc = self.pos_coverage(pred)
        nc = self.neg_coverage(pred)

        pred.set_scores(pc, nc)

        return pc >= 1.0 and nc <= 0.0


def new_task(pos, neg):
    return Task(pos, neg)
