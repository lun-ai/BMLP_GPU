import pygraphblas as gb
import unittest
import bmlp.matrix


class BMLPTests(unittest.TestCase):

    def test_load_pl(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = bmlp.matrix.integers_to_boolean_matrix("bmlp/load_pl_test.pl")

        self.assertEqual(a[0, 1], True)
        self.assertEqual(a[1, 2], True)
        self.assertEqual(a[2, 3], True)
        self.assertEqual(a[3, 0], True)
        self.assertEqual(a[4, 4], True)

    def test_load_pl_squared(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = bmlp.matrix.integers_to_boolean_matrix(
            "bmlp/load_pl_test.pl", is_squared=True)

        self.assertEqual(a[0, 1], True)
        self.assertEqual(a[1, 2], True)
        self.assertEqual(a[2, 3], True)
        self.assertEqual(a[3, 0], True)
        self.assertEqual(a[4, 4], True)
        self.assertEqual(a.nrows, 6)

    def test_save_pl(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = bmlp.matrix.integers_to_boolean_matrix("bmlp/load_pl_test.pl")

        # Create a copy of this boolean matrix
        b = a

        # Save this copy into a .pl file and rename the predicate
        bmlp.matrix.boolean_matrix_to_integers(b, "b", "bmlp/output.pl")

    def test_simple_chain(self):

        # Create a sequence of boolean matrix operations to compute
        #   p0(X1, X2) :- p1(X1, X3), p2(X3, X4), p3(X4, X2).

        num_nodes = 5

        p1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        p2 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        p3 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        p1[0, 1] = True
        p2[1, 2] = True
        p3[2, 3] = True

        # exactly-two-connected program in chained H2m
        p0 = p1 @ p2 @ p3

        self.assertEqual(p0[0, 3], True)

    def test_bmlp_rms_simple_recursion(self):

        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        num_nodes = 5

        # Create a square adjacency matrix using BOOL type
        R1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = True

        res = bmlp.matrix.BMLP_RMS(R1)

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
        R1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = True

        res = bmlp.matrix.BMLP_RMS(R1, R1)

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
        R1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        R2 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges[1:]:
            R1[src, dst] = True

        for src, dst in edges:
            R2[src, dst] = True

        res = bmlp.matrix.BMLP_RMS(R1, R2)

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
        p1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        p2 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        p3 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        p1[1, 0] = True

        p2[1, 2] = True
        p2[3, 4] = True

        p3[2, 3] = True
        p3[4, 0] = True

        # exactly-two-connected recursion in chained H2m
        res = bmlp.matrix.BMLP_RMS(p1.T, p2 @ p3)

        self.assertEqual(res[0, 1], True)
        self.assertEqual(res[1, 1], True)
        self.assertEqual(res[3, 1], True)

    def test_bmlp_smp_simple_recursion(self):

        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 4)]
        num_nodes = 5

        # Create a square adjacency matrix using BOOL type
        R1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = True

        # Create a vector to represent a query
        V = gb.Vector.sparse(gb.BOOL, num_nodes)

        # query the reachability of node 3
        V[3] = True

        res = bmlp.matrix.BMLP_SMP(V, R1)

        self.assertEqual(res[0], True)
        self.assertEqual(res[1], True)
        self.assertEqual(res[2], True)
        self.assertEqual(res[3], True)

    def test_bmlp_ie(self):

        edges_1 = [(0, 0), (1, 1), (2, 2), (2, 3)]
        edges_2 = [(0, 1), (1, 2), (1, 3), (2, 4)]

        num_rows = 3
        num_cols = 5

        # Create the first matrix using BOOL type
        R1 = gb.Matrix.sparse(gb.BOOL, num_rows, num_cols)

        # Create the second matrix using BOOL type
        R2 = gb.Matrix.sparse(gb.BOOL, num_rows, num_cols)

        # Insert edges into the first matrix
        for src, dst in edges_1:
            R1[src, dst] = True

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = True

        # Create a vector to represent a query
        V = gb.Matrix.sparse(gb.BOOL, 2, num_cols)

        # query the reachability of node 1
        V[0, 0] = True
        V[1, 1] = True

        res, _ = bmlp.matrix.BMLP_IE(V, R1, R2)

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
        R1 = gb.Matrix.sparse(gb.BOOL, num_rows, num_cols)

        # Create the second matrix using BOOL type
        R2 = gb.Matrix.sparse(gb.BOOL, num_rows, num_cols)

        # Insert edges into the first matrix
        for src, dst in edges_1:
            R1[src, dst] = True

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = True

        # Create a vector to represent a query
        V = gb.Matrix.sparse(gb.BOOL, 2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = gb.Matrix.sparse(gb.BOOL, 2, num_rows)

        # Query the reachability of node 0 and 3
        V[0, 0] = True
        V[0, 2] = True
        V[0, 3] = True
        # Filter the second row in the matrix
        T[0, 0] = True
        T[0, 2] = True

        res, _ = bmlp.matrix.BMLP_IE(V, R1, R2, T)

        self.assertEqual(res[0, 0], True)
        self.assertEqual(res[0, 1], False)
        self.assertEqual(res[0, 2], True)
        self.assertEqual(res[0, 3], True)
        self.assertEqual(res[0, 4], False)

    def test_bmlp_ie_filtering_2(self):

        edges_1 = [(0, 0), (1, 1), (2, 2), (2, 3)]
        edges_2 = [(0, 1), (1, 2), (1, 3), (2, 4)]

        num_rows = 3
        num_cols = 5

        # Create the first matrix using BOOL type
        R1 = gb.Matrix.sparse(gb.BOOL, num_rows, num_cols)

        # Create the second matrix using BOOL type
        R2 = gb.Matrix.sparse(gb.BOOL, num_rows, num_cols)

        # Insert edges into the first matrix
        for src, dst in edges_1:
            R1[src, dst] = True

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = True

        # Create a vector to represent a query
        V = gb.Matrix.sparse(gb.BOOL, 2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = gb.Matrix.sparse(gb.BOOL, 2, num_rows)

        # Query the reachability of node 0 and 3
        V[0, 0] = True
        V[0, 3] = True
        # Filter the second row in the matrix
        T[0, 1] = True

        # Query the reachability of node 1
        V[1, 1] = True
        # Filter the first row in the matrix
        T[1, 0] = True

        res, _ = bmlp.matrix.BMLP_IE(V, R1, R2, T)

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
        R1 = gb.Matrix.sparse(gb.BOOL, num_rows, num_cols)

        # Create the second matrix using BOOL type
        R2 = gb.Matrix.sparse(gb.BOOL, num_rows, num_cols)

        # Insert edges into the first matrix
        for src, dst in edges_1:
            R1[src, dst] = True

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = True

        # Create a vector to represent a query
        V = gb.Matrix.sparse(gb.BOOL, 2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = gb.Matrix.sparse(gb.BOOL, 2, num_rows)

        # Node 0 should reach all other nodes
        V[0, 0] = True

        # Now node 0 should not reach any other node
        V[1, 0] = True
        # Filter the first row in the matrix
        T[1, 0] = True
        T[1, 1] = True
        T[1, 2] = True

        res, _ = bmlp.matrix.BMLP_IE(V, R1, R2, T)

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
