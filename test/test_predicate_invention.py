import unittest
from bmlp.core import matrix
from bmlp.learn import predicate, task, generator, learn


class TestBMLPPredicateInvention(unittest.TestCase):

    def test_bmlp_ILP_two_body(self):
        #######################################################
        # Background knowledge:
        #
        # harry+sally
        # /		\
        # john		mary
        #
        # father(harry,john). mother(sally,john).
        # father(harry,mary). mother(sally,mary).
        # male(harry). female(sally).
        # male(john). female(mary).
        #
        # Constant to matrix index mapping:
        #   (0, harry), (1, john), (2, mary), (3, sally)
        #
        # Target:
        #   parent
        #
        # Examples:
        #   E+ =
        #       {   parent(harry,john). parent(sally,john).
        #           parent(harry,mary). parent(sally,mary). }
        #   E- =
        #       {   parent(harry,sally). parent(mary,john).
        #           parent(harry,harry).                    }

        m1 = matrix.new(64, 64)
        m1[0, 1] = True
        m1[0, 2] = True
        father = predicate.new_predicate(m1, "father")

        m2 = matrix.new(64, 64)
        m2[3, 1] = True
        m2[3, 2] = True
        mother = predicate.new_predicate(m2, "mother")

        m3 = matrix.new(64, 64)
        m3[0, 0] = True
        m3[1, 1] = True
        male = predicate.new_predicate(m3, "male")

        m4 = matrix.new(64, 64)
        m4[2, 2] = True
        m4[3, 3] = True
        female = predicate.new_predicate(m4, "female")

        pos = matrix.new(64, 64)
        pos[0, 1] = True
        pos[0, 2] = True
        pos[3, 1] = True
        pos[3, 2] = True
        # Negative examples
        neg = matrix.new(64, 64)
        neg[0, 3] = True
        neg[2, 1] = True
        neg[0, 0] = True

        learn_parent = task.new_task(pos, neg)

        # Primitive predicates are the depth 1 predicates
        primitives = [father, mother, male, female]

        # Define a generator using default operators
        gen = generator.Generator(primitives)
        success, prog, _ = gen.invent_predicates_for_task(learn_parent)

        self.assertTrue(success)
        # self.assertEqual(
        # str(res[0]), "inv_14(X, Y) :- father(X, Y).\ninv_14(X, Y) :- mother(X, Y).\n")
        self.assertEqual(prog[0].get_scores(), (1.0, 0.0))

    def test_bmlp_ILP_PI(self):
        #######################################################
        # Background knowledge:
        #
        #               harry+sally
        #               /		\
        #             john		mary
        #             |          |
        #           bill        maggie
        # father(harry,john). mother(sally,john).
        # father(harry,mary). mother(sally,mary).
        # father(john, bill). mother(mary, maggie).
        # male(harry). female(sally).
        # male(john). female(mary).
        # male(bill). female(maggie).
        #
        # Constant to matrix index mapping:
        #   (0, harry), (1, john), (2, mary), (3, sally), (4, bill), (5, maggie)
        #
        #
        # Target:
        #   grandparent
        #
        # Examples:
        #   E+ =
        #       {   grandparent(harry,bill). grandparent(sally,bill).
        #           grandparent(harry,maggie). grandparent(sally,maggie). }
        #   E- =
        #       {   grandparent(harry,sally). grandparent(mary,john).
        #           grandparent(harry,harry). grandparent(john, bill).  }

        m1 = matrix.new(64, 64)
        m1[0, 1] = True
        m1[0, 2] = True
        m1[1, 4] = True
        father = predicate.new_predicate(m1, "father")

        m2 = matrix.new(64, 64)
        m2[3, 1] = True
        m2[3, 2] = True
        m2[2, 5] = True
        mother = predicate.new_predicate(m2, "mother")

        m3 = matrix.new(64, 64)
        m3[0, 0] = True
        m3[1, 1] = True
        m3[4, 4] = True
        male = predicate.new_predicate(m3, "male")

        m4 = matrix.new(64, 64)
        m4[2, 2] = True
        m4[3, 3] = True
        m4[5, 5] = True
        female = predicate.new_predicate(m4, "female")

        # Postive examples
        pos = matrix.new(64, 64)
        pos[0, 4] = True
        pos[0, 5] = True
        pos[3, 4] = True
        pos[3, 5] = True

        # Negative examples
        neg = matrix.new(64, 64)
        neg[0, 3] = True
        neg[2, 1] = True
        neg[0, 0] = True
        neg[1, 4] = True

        learn_grandparent = task.new_task(pos, neg)

        primitives = [father, mother, male, female]
        gen = generator.Generator(primitives)
        cached = {}

        # Generate predicates
        success, _, size_2 = gen.invent_predicates_for_task(
            learn_grandparent, cached)

        # Generate more predicates
        gen.update_predicates(len(size_2), size_2 + primitives)
        success, prog, _ = gen.invent_predicates_for_task(
            learn_grandparent, cached)

        self.assertTrue(success)
        # self.assertEqual(str(res[0]),
        #                  "inv_14(X, Y) :- father(X, Y).\ninv_14(X, Y) :- mother(X, Y).\ninv_78(X, Y) :- inv_14(X, Z), inv_14(Z, Y).\n")
        self.assertEqual(prog[0].get_scores(), (1.0, 0.0))

    def test_bmlp_ILP_two_body_learning_1(self):
        #######################################################
        # Background knowledge:
        #
        #               harry+sally
        #               /		\
        #             john		mary
        #             |          |
        #           bill        maggie
        # father(harry,john). mother(sally,john).
        # father(harry,mary). mother(sally,mary).
        # father(john, bill). mother(mary, maggie).
        # male(harry). female(sally).
        # male(john). female(mary).
        # male(bill). female(maggie).
        #
        # Constant to matrix index mapping:
        #   (0, harry), (1, john), (2, mary), (3, sally), (4, bill), (5, maggie)
        #
        #
        # Target:
        #   grandparent
        #
        # Examples:
        #   E+ =
        #       {   grandparent(harry,bill). grandparent(sally,bill).
        #           grandparent(harry,maggie). grandparent(sally,maggie). }
        #   E- =
        #       {   grandparent(harry,sally). grandparent(mary,john).
        #           grandparent(harry,harry). grandparent(john, bill).  }

        m1 = matrix.new(64, 64)
        m1[0, 1] = True
        m1[0, 2] = True
        m1[1, 4] = True
        father = predicate.new_predicate(m1, "father")

        m2 = matrix.new(64, 64)
        m2[3, 1] = True
        m2[3, 2] = True
        m2[2, 5] = True
        mother = predicate.new_predicate(m2, "mother")

        m3 = matrix.new(64, 64)
        m3[0, 0] = True
        m3[1, 1] = True
        m3[4, 4] = True
        male = predicate.new_predicate(m3, "male")

        m4 = matrix.new(64, 64)
        m4[2, 2] = True
        m4[3, 3] = True
        m4[5, 5] = True
        female = predicate.new_predicate(m4, "female")

        # Postive examples
        pos = matrix.new(64, 64)
        pos[0, 4] = True
        pos[0, 5] = True
        pos[3, 4] = True
        pos[3, 5] = True

        # Negative examples
        neg = matrix.new(64, 64)
        neg[0, 3] = True
        neg[2, 1] = True
        neg[0, 0] = True
        neg[1, 4] = True

        primitives = [father, mother, male, female]

        prog = learn.learn_task(pos, neg, primitives, verbose=False)

        self.assertEqual(prog[0].get_scores(), (1.0, 0.0))

    def test_bmlp_ILP_two_body_learning_2(self):
        #######################################################
        # Background knowledge:
        #
        # harry+sally
        # /		\
        # john		mary
        #             |          |
        #           bill        maggie
        #           \
        #           ted
        #
        # father(harry,john). mother(sally,john).
        # father(harry,mary). mother(sally,mary).
        # father(john, bill). mother(mary, maggie).
        # father(bill, ted).
        # male(harry). female(sally).
        # male(john). female(mary).
        # male(bill). female(maggie).
        # male(ted).
        #
        # Constant to matrix index mapping:
        #   (0, harry), (1, john), (2, mary), (3, sally), (4, bill), (5, maggie), (6, ted)
        #
        #
        # Target:
        #   ancestor
        #
        # Examples:
        #   E+ =
        #       {
        #           ancestor(harry,ted).  ancestor(harry,john).
        #           ancestor(harry,bill). ancestor(sally,bill).
        #           ancestor(harry,maggie). ancestor(sally,maggie). }
        #   E- =
        #       {   ancestor(harry,sally). ancestor(mary,john).
        #           ancestor(harry,harry). ancestor(ted, bill).  }
        m1 = matrix.new(64, 64)
        m1[0, 1] = True
        m1[0, 2] = True
        m1[1, 4] = True
        m1[4, 6] = True
        father = predicate.new_predicate(m1, "father")

        m2 = matrix.new(64, 64)
        m2[3, 1] = True
        m2[3, 2] = True
        m2[2, 5] = True
        mother = predicate.new_predicate(m2, "mother")

        m3 = matrix.new(64, 64)
        m3[0, 0] = True
        m3[1, 1] = True
        m3[4, 4] = True
        male = predicate.new_predicate(m3, "male")

        m4 = matrix.new(64, 64)
        m4[2, 2] = True
        m4[3, 3] = True
        m4[5, 5] = True
        female = predicate.new_predicate(m4, "female")

        # Postive examples
        pos = matrix.new(64, 64)
        pos[0, 1] = True
        pos[0, 4] = True
        pos[0, 5] = True
        pos[0, 6] = True
        pos[3, 4] = True
        pos[3, 5] = True

        # Negative examples
        neg = matrix.new(64, 64)
        neg[0, 3] = True
        neg[2, 1] = True
        neg[0, 0] = True
        neg[6, 4] = True

        primitives = [father, mother, male, female]

        prog = learn.learn_task(pos, neg, primitives, verbose=False)
        self.assertEqual(prog[0].get_scores(), (1.0, 0.0))
