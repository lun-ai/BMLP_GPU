import unittest
from bmlp.core.matrix import *
from bmlp.core.utils import *
import os

# Get absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get directory containing the current file
current_dir = os.path.dirname(current_file_path)


class TestCoreMatrix(unittest.TestCase):

    def test_mul(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)
        b = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        b[1, 2] = True

        c = mul(a, b)

        self.assertEqual(c[0, 2], True)
        self.assertNotEqual(c.get(1, 2), True)
        self.assertNotEqual(c.get(0, 1), True)

    def test_mul_inplace(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)
        b = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        b[1, 2] = True

        c = new(3, 3)
        self.assertNotEqual(c.get(0, 2), True)

        c << mul(a, b, inplace=True)

        self.assertEqual(c[0, 2], True)
        self.assertNotEqual(c.get(1, 2), True)
        self.assertNotEqual(c.get(0, 1), True)

    def test_square(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        a[1, 2] = True

        b = square(a)

        self.assertEqual(b[0, 2], True)
        self.assertNotEqual(b.get(1, 2), True)
        self.assertNotEqual(b.get(0, 1), True)

    def test_square_inplace(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        a[1, 2] = True

        a << square(a, inplace=True)

        self.assertEqual(a[0, 2], True)
        self.assertNotEqual(a.get(1, 2), True)
        self.assertNotEqual(a.get(0, 1), True)

    def test_add(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)
        b = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        b[0, 1] = True
        b[1, 2] = True

        c = add(a, b)

        self.assertEqual(c[0, 1], True)
        self.assertEqual(c[1, 2], True)

    def test_add_inplace(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)
        b = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        b[0, 1] = True
        b[1, 2] = True

        c = new(3, 3)
        c << add(a, b, inplace=True)

        self.assertEqual(c[0, 1], True)
        self.assertEqual(c[1, 2], True)

    def test_intersection(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)
        b = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        b[0, 1] = True
        b[1, 2] = True

        c = intersection(a, b)

        self.assertEqual(c[0, 1], True)
        self.assertNotEqual(c.get(1, 2), True)

    def test_intersection_inplace(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)
        b = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        b[0, 1] = True
        b[1, 2] = True

        c = new(3, 3)
        c << intersection(a, b, inplace=True)

        self.assertEqual(c[0, 1], True)
        self.assertNotEqual(c.get(1, 2), True)

    def test_transpose(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        a[1, 2] = True

        b = transpose(a)

        self.assertEqual(b[1, 0], True)
        self.assertEqual(b[2, 1], True)
        self.assertNotEqual(b[0, 1], True)
        self.assertNotEqual(b[1, 2], True)

    def test_transpose_inplace(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        a[1, 2] = True

        a << transpose(a, inplace=True)

        self.assertEqual(a[1, 0], True)
        self.assertEqual(a[2, 1], True)
        self.assertNotEqual(a[0, 1], True)
        self.assertNotEqual(a[1, 2], True)

    def test_negate(self):

        # Create a square adjacency matrix using BOOL type
        a = new(2, 2)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        a[1, 1] = True

        b = negate(a)

        self.assertNotEqual(b[0, 1], True)
        self.assertNotEqual(b[1, 1], True)
        self.assertEqual(b[0, 0], True)
        self.assertEqual(b[1, 0], True)

    def test_negate_inplace(self):

        # Create a square adjacency matrix using BOOL type
        a = new(2, 2)

        # Insert edges into the adjacency matrix
        a[0, 1] = True
        a[1, 1] = True

        a << negate(a, inplace=True)

        self.assertNotEqual(a[0, 1], True)
        self.assertNotEqual(a[1, 1], True)
        self.assertEqual(a[0, 0], True)
        self.assertEqual(a[1, 0], True)

    def test_identity(self):

        b = identity(2)

        self.assertEqual(b[0, 0], True)
        self.assertEqual(b[1, 1], True)
        self.assertNotEqual(b[0, 1], True)
        self.assertNotEqual(b[1, 0], True)

    def test_resize(self):

        # Create a square adjacency matrix using BOOL type
        a = new(2, 2)

        # Insert edges into the adjacency matrix
        a[0, 1] = True

        b = resize(a, 6, 6)

        b[5, 5] = True

        self.assertEqual(b[0, 1], True)
        self.assertNotEqual(b[1, 2], True)
        self.assertNotEqual(b[2, 3], True)
        self.assertNotEqual(b[3, 0], True)
        self.assertNotEqual(b[4, 4], True)
        self.assertEqual(b[5, 5], True)

    def test_resize_inplace(self):

        # Create a square adjacency matrix using BOOL type
        a = new(2, 2)

        # Insert edges into the adjacency matrix
        a[0, 1] = True

        a << resize(a, 6, 6, inplace=True)

        a[5, 5] = True

        self.assertEqual(a[0, 1], True)
        self.assertNotEqual(a[1, 2], True)
        self.assertNotEqual(a[2, 3], True)
        self.assertNotEqual(a[3, 0], True)
        self.assertNotEqual(a[4, 4], True)
        self.assertEqual(a[5, 5], True)


class TestIO(unittest.TestCase):

    def test_create_matrix(self):

        # Create a square adjacency matrix using BOOL type
        a = new(5, 5)

        # Insert edges into the adjacency matrix
        a[0, 1] << True
        a[1, 2] << True
        a[2, 3] << True
        a[3, 0] << True
        a[4, 4] << True

        self.assertEqual(a[0, 1], True)
        self.assertEqual(a[1, 2], True)
        self.assertEqual(a[2, 3], True)
        self.assertEqual(a[3, 0], True)
        self.assertEqual(a[4, 4], True)

    def test_save_matrix(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] << True
        a[1, 2] << True

        save(a, current_dir + "/output.csv")

    def test_load_matrix(self):

        # Create a square adjacency matrix using BOOL type
        a = new(3, 3)

        # Insert edges into the adjacency matrix
        a[0, 1] << True
        a[1, 2] << True

        save(a, current_dir + "/output.csv")
        b = load(current_dir + "/output.csv")

        self.assertEqual(b[0, 1], True)
        self.assertEqual(b[1, 2], True)
        self.assertNotEqual(b[0, 0], True)
        self.assertNotEqual(b[0, 2], True)
        self.assertNotEqual(b[1, 1], True)
        self.assertNotEqual(b[1, 0], True)
        self.assertNotEqual(b[2, 0], True)
        self.assertNotEqual(b[2, 1], True)
        self.assertNotEqual(b[2, 2], True)

    def test_load_pl(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix(current_dir + "/int_matrix.pl")

        self.assertEqual(a[0, 1], True)
        self.assertEqual(a[1, 2], True)
        self.assertEqual(a[2, 3], True)
        self.assertEqual(a[3, 0], True)
        self.assertEqual(a[4, 4], True)

    def test_load_pl_squared(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix(
            current_dir + "/int_matrix.pl", is_squared=True)

        self.assertEqual(a[0, 1], True)
        self.assertEqual(a[1, 2], True)
        self.assertEqual(a[2, 3], True)
        self.assertEqual(a[3, 0], True)
        self.assertEqual(a[4, 4], True)
        self.assertEqual(a.nrows, 5)

    def test_save_pl(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix(current_dir + "/int_matrix.pl")

        # Create a copy of this boolean matrix
        b = a

        # Save this copy into a .pl file and rename the predicate
        boolean_matrix_to_integers(b, "b", current_dir + "/output1.pl")

    def test_run_and_save_pl(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix(current_dir + "/int_matrix.pl")

        # Create a copy of this boolean matrix
        b = RMS(a)

        # Save this copy into a .pl file and rename the predicate
        boolean_matrix_to_integers(b, "b", current_dir + "/output2.pl")


class TestBMLPModules(unittest.TestCase):

    def test_simple_chain(self):

        # Create a sequence of boolean matrix operations to compute
        #   p0(X1, X2) :- p1(X1, X3), p2(X3, X4), p3(X4, X2).

        num_nodes = 5

        p1 = new(num_nodes, num_nodes)
        p2 = new(num_nodes, num_nodes)
        p3 = new(num_nodes, num_nodes)

        p1[0, 1] = True
        p2[1, 2] = True
        p3[2, 3] = True

        # exactly-two-connected program in chained H2m
        p0 = mul(mul(p1, p2), p3)

        self.assertEqual(p0[0, 3], True)

    def test_bmlp_rms_simple_recursion(self):

        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        num_nodes = 5

        # Create a square adjacency matrix using BOOL type
        R1 = new(num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = True

        res = RMS(R1)

        self.assertEqual(res[0, 1], True)
        self.assertEqual(res[0, 2], True)
        self.assertEqual(res[0, 3], True)
        self.assertEqual(res[0, 4], True)
        self.assertEqual(res[1, 2], True)
        self.assertEqual(res[1, 3], True)
        self.assertEqual(res[1, 4], True)
        self.assertEqual(res[2, 3], True)
        self.assertEqual(res[2, 4], True)
        self.assertEqual(res[3, 4], True)

    def test_bmlp_rms_simple_recursion_with_same_body(self):

        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        num_nodes = 5

        # Create a square adjacency matrix using BOOL type
        R1 = new(num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = True

        res = RMS(R1, R1)

        self.assertEqual(res[0, 1], True)
        self.assertEqual(res[0, 2], True)
        self.assertEqual(res[0, 3], True)
        self.assertEqual(res[0, 4], True)
        self.assertEqual(res[1, 2], True)
        self.assertEqual(res[1, 3], True)
        self.assertEqual(res[1, 4], True)
        self.assertEqual(res[2, 3], True)
        self.assertEqual(res[2, 4], True)
        self.assertEqual(res[3, 4], True)

    def test_bmlp_rms_simple_recursion_with_diff_body_1(self):

        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        num_nodes = 5

        # Create a square adjacency matrix using BOOL type
        R1 = new(num_nodes, num_nodes)
        R2 = new(num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges[1:]:
            R1[src, dst] = True

        for src, dst in edges:
            R2[src, dst] = True

        res = RMS(R1, R2)

        self.assertEqual(res[0, 2], True)
        self.assertEqual(res[0, 3], True)
        self.assertEqual(res[0, 4], True)
        self.assertEqual(res[1, 2], True)
        self.assertEqual(res[1, 3], True)
        self.assertEqual(res[1, 4], True)
        self.assertEqual(res[2, 3], True)
        self.assertEqual(res[2, 4], True)
        self.assertEqual(res[3, 4], True)

    def test_bmlp_rms_simple_recursion_with_diff_body_2(self):

        # Create a sequence of boolean matrix operations to compute
        #   p0(X1, X2) :- p1(X2, X1).
        #   p0(X1, X2) :- p2(X1, X3), p3(X3, X4), p0(X4, X2).
        num_nodes = 5
        p1 = new(num_nodes, num_nodes)
        p2 = new(num_nodes, num_nodes)
        p3 = new(num_nodes, num_nodes)

        p1[1, 0] = True

        p2[1, 2] = True
        p2[3, 4] = True

        p3[2, 3] = True
        p3[4, 0] = True

        # exactly-two-connected recursion in chained H2m
        res = RMS(transpose(p1), mul(p2, p3))

        self.assertEqual(res[0, 1], True)
        self.assertEqual(res[1, 1], True)
        self.assertEqual(res[3, 1], True)

    def test_bmlp_smp_simple_recursion(self):

        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 4)]
        num_nodes = 5

        # Create a square adjacency matrix using BOOL type
        R1 = new(num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = True

        # Create a vector to represent a query
        V = new(num_nodes)

        # query the reachability of node 3
        V[3] = True

        res = SMP(V, R1)

        self.assertEqual(res[0], True)
        self.assertEqual(res[1], True)
        self.assertEqual(res[2], True)
        self.assertEqual(res[3], True)
        self.assertNotEqual(res[4], True)

    def test_bmlp_ie(self):

        edges_1 = [(0, 0), (1, 1), (2, 2), (2, 3)]
        edges_2 = [(0, 1), (1, 2), (1, 3), (2, 4)]

        num_rows = 3
        num_cols = 5

        # Create the first matrix using BOOL type
        R1 = new(num_rows, num_cols)

        # Create the second matrix using BOOL type
        R2 = new(num_rows, num_cols)

        # Insert edges into the first matrix
        for src, dst in edges_1:
            R1[src, dst] = True

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = True

        # Create a vector to represent a query
        V = new(2, num_cols)

        # query the reachability of node 1
        V[0, 0] = True
        V[1, 1] = True

        res, _ = IE(V, R1, R2)

        self.assertEqual(res[0, 0], True)
        self.assertEqual(res[0, 1], True)
        self.assertEqual(res[0, 2], True)
        self.assertEqual(res[0, 3], True)
        self.assertEqual(res[1, 1], True)
        self.assertEqual(res[1, 2], True)
        self.assertEqual(res[1, 3], True)

    def test_bmlp_ie_filtering_1(self):

        edges_1 = [(0, 0), (1, 1), (2, 2), (2, 3)]
        edges_2 = [(0, 1), (1, 2), (1, 3), (2, 4)]

        num_rows = 3
        num_cols = 5

        # Create the first matrix using BOOL type
        R1 = new(num_rows, num_cols)

        # Create the second matrix using BOOL type
        R2 = new(num_rows, num_cols)

        # Insert edges into the first matrix
        for src, dst in edges_1:
            R1[src, dst] = True

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = True

        # Create a vector to represent a query
        V = new(2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = new(2, num_rows)

        # Query the reachability of node 0 and 3
        V[0, 0] = True
        V[0, 2] = True
        V[0, 3] = True
        # Filter the second row in the matrix
        T[0, 0] = True
        T[0, 2] = True

        res, _ = IE(V, R1, R2, T)

        self.assertEqual(res[0, 0], True)
        self.assertNotEqual(res.get(0, 1), True)
        self.assertEqual(res[0, 2], True)
        self.assertEqual(res[0, 3], True)
        self.assertNotEqual(res.get(0, 4), True)

    def test_bmlp_ie_filtering_2(self):

        edges_1 = [(0, 0), (1, 1), (2, 2), (2, 3)]
        edges_2 = [(0, 1), (1, 2), (1, 3), (2, 4)]

        num_rows = 3
        num_cols = 5

        # Create the first matrix using BOOL type
        R1 = new(num_rows, num_cols)

        # Create the second matrix using BOOL type
        R2 = new(num_rows, num_cols)

        # Insert edges into the first matrix
        for src, dst in edges_1:
            R1[src, dst] = True

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = True

        # Create a vector to represent a query
        V = new(2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = new(2, num_rows)

        # Query the reachability of node 0 and 3
        V[0, 0] = True
        V[0, 3] = True
        # Filter the second row in the matrix
        T[0, 1] = True

        # Query the reachability of node 1
        V[1, 1] = True
        # Filter the first row in the matrix
        T[1, 0] = True

        res, _ = IE(V, R1, R2, T)

        self.assertEqual(res[0, 0], True)
        self.assertEqual(res[0, 1], True)
        self.assertNotEqual(res.get(0, 2), True)
        self.assertEqual(res[0, 3], True)
        self.assertNotEqual(res.get(0, 4), True)

        self.assertNotEqual(res.get(1, 0), True)
        self.assertEqual(res[1, 1], True)
        self.assertEqual(res[1, 2], True)
        self.assertEqual(res[1, 3], True)
        self.assertEqual(res[1, 4], True)

    def test_bmlp_ie_empty_R1(self):

        # No requirements from R1
        edges_1 = []
        edges_2 = [(0, 1), (1, 2), (1, 3), (2, 4)]

        num_rows = 3
        num_cols = 5

        # Create the first matrix using BOOL type
        R1 = new(num_rows, num_cols)

        # Create the second matrix using BOOL type
        R2 = new(num_rows, num_cols)

        # Insert edges into the first matrix
        for src, dst in edges_1:
            R1[src, dst] = True

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = True

        # Create a vector to represent a query
        V = new(2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = new(2, num_rows)

        # Node 0 should reach all other nodes
        V[0, 0] = True

        # Filter the first row in the matrix
        T[1, 0] = True
        T[1, 1] = True
        T[1, 2] = True
        # Now node 0 should not reach any other node
        V[1, 0] = True

        res, _ = IE(V, R1, R2, T)

        self.assertEqual(res[0, 0], True)
        self.assertEqual(res[0, 1], True)
        self.assertEqual(res[0, 2], True)
        self.assertEqual(res[0, 3], True)
        self.assertEqual(res[0, 4], True)

        self.assertEqual(res[1, 0], True)
        self.assertNotEqual(res.get(1, 1), True)
        self.assertNotEqual(res.get(1, 2), True)
        self.assertNotEqual(res.get(1, 3), True)
        self.assertNotEqual(res.get(1, 4), True)


# Testsuite treated as if a top-level module
if __name__ == '__main__':
    unittest.main()
