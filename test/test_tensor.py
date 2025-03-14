import unittest
import torch
import numpy as np
import os
from bmlp.core.tensor import *
from bmlp.core.utils import *


# Get absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get directory containing the current file
current_dir = os.path.dirname(current_file_path)


class TestCoreTorchMatrix(unittest.TestCase):

    def test_mul(self):
        # Create a square adjacency matrix
        a = torch.tensor(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )
        b = torch.tensor(
            [[0.0, 0.1, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )

        c = mul(a, b)

        # Test with to_dense() for easier comparison
        self.assertTrue(b[0, 1].item())
        self.assertTrue(c[0, 2].item())
        self.assertFalse(c[1, 2].item())
        self.assertFalse(c[0, 1].item())

    def test_square(self):
        # Create a square adjacency matrix
        a = torch.tensor(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )

        b = square(a)

        self.assertTrue(b[0, 2].item())
        self.assertFalse(b[1, 2].item())
        self.assertFalse(b[0, 1].item())

    def test_add(self):
        # Create square adjacency matrices
        a = torch.tensor(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )
        b = torch.tensor(
            [[0.0, 1.0, 1.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )

        c = add(a, b)

        self.assertTrue(c[0, 1].item())
        self.assertTrue(c[0, 2].item())
        self.assertTrue(c[1, 2].item())
        self.assertFalse(c[0, 0].item())
        self.assertFalse(c[1, 0].item())
        self.assertFalse(c[1, 1].item())

    def test_intersection(self):
        # Create square adjacency matrices
        a = torch.tensor(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )
        b = torch.tensor(
            [[0.0, 1.0, 1.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )

        c = intersection(a, b)

        self.assertTrue(c[0, 1])
        self.assertTrue(c[1, 2])
        self.assertFalse(c[0, 2])

    def test_transpose(self):
        # Create a square adjacency matrix
        a_indices = torch.tensor([[0, 1], [1, 2]], dtype=D_TYPE)
        a_values = torch.tensor([1.0, 1.0], dtype=D_TYPE)
        a = torch.sparse_coo_tensor(a_indices, a_values, (3, 3))

        b = transpose(a)

        self.assertTrue(b[1, 0].item())
        self.assertTrue(b[2, 1].item())
        self.assertFalse(b[0, 1].item())
        self.assertFalse(b[1, 2].item())

    def test_negate(self):
        # Create a square adjacency matrix
        a = torch.tensor(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )

        b = negate(a)

        self.assertFalse(b[0, 1].item())
        self.assertFalse(b[1, 2].item())
        self.assertTrue(b[0, 0].item())
        self.assertTrue(b[0, 2].item())
        self.assertTrue(b[1, 0].item())
        self.assertTrue(b[1, 1].item())
        self.assertTrue(b[2, 0].item())
        self.assertTrue(b[2, 1].item())
        self.assertTrue(b[2, 2].item())

    def test_identity(self):
        b = identity(2)

        self.assertTrue(b[0, 0].item())
        self.assertTrue(b[1, 1].item())
        self.assertFalse(b[0, 1].item())
        self.assertFalse(b[1, 0].item())

    def test_resize(self):
        # Create a square adjacency matrix
        a = torch.tensor(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0]]
        )

        b = resize(a, 6, 6)
        b[5, 5] = 1.0

        self.assertTrue(b[0, 1].item())
        self.assertTrue(b[1, 2].item())
        self.assertFalse(b[0, 0].item())
        self.assertFalse(b[0, 2].item())
        self.assertFalse(b[1, 0].item())
        self.assertFalse(b[1, 1].item())
        self.assertFalse(b[2, 3].item())
        self.assertFalse(b[3, 0].item())
        self.assertFalse(b[4, 4].item())
        self.assertTrue(b[5, 5].item())


class TestIOTorch(unittest.TestCase):

    def test_save_load_matrix(self):
        # Create a square adjacency matrix
        a = torch.tensor([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, 0.0]])

        # Test save functionality
        save(a, current_dir + "/output_torch.csv")

        # Test load functionality
        b = load(current_dir + "/output_torch.csv")

        self.assertTrue(b[0, 1].item())
        self.assertTrue(b[1, 2].item())
        self.assertFalse(b[0, 0].item())
        self.assertFalse(b[0, 2].item())

    def test_create_matrix(self):

        # Create a square adjacency matrix using BOOL type
        a = new(5, 5)

        # Insert edges into the adjacency matrix
        a[0, 1] = 1.0
        a[1, 2] = 1.0
        a[2, 3] = 1.0
        a[3, 0] = 1.0
        a[4, 4] = 1.0

        self.assertEqual(a[0, 1], 1.0)
        self.assertEqual(a[1, 2], 1.0)
        self.assertEqual(a[2, 3], 1.0)
        self.assertEqual(a[3, 0], 1.0)
        self.assertEqual(a[4, 4], 1.0)

    def test_load_pl(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix(
            current_dir + "/int_matrix.pl", to_tensor=True)

        self.assertIsInstance(a, torch.Tensor)
        self.assertEqual(a[0, 1], 1.0)
        self.assertEqual(a[1, 2], 1.0)
        self.assertEqual(a[2, 3], 1.0)
        self.assertEqual(a[3, 0], 1.0)
        self.assertEqual(a[4, 4], 1.0)

    def test_load_pl_squared(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix(
            current_dir + "/int_matrix.pl", is_squared=True, to_tensor=True)

        self.assertIsInstance(a, torch.Tensor)
        self.assertEqual(a[0, 1], 1.0)
        self.assertEqual(a[1, 2], 1.0)
        self.assertEqual(a[2, 3], 1.0)
        self.assertEqual(a[3, 0], 1.0)
        self.assertEqual(a[4, 4], 1.0)
        self.assertEqual(a.shape[0], 5)
        self.assertEqual(a.shape[1], 5)

    def test_save_pl(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix(
            current_dir + "/int_matrix.pl", to_tensor=True)

        # Create a copy of this boolean matrix
        self.assertIsInstance(a, torch.Tensor)
        b = a

        # Save this copy into a .pl file and rename the predicate
        boolean_matrix_to_integers(b, "b", current_dir + "/output1.pl")

    def test_run_and_save_pl(self):

        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix(
            current_dir + "/int_matrix.pl", to_tensor=True)

        self.assertIsInstance(a, torch.Tensor)

        # Create a copy of this boolean matrix
        b = RMS(a)

        # Save this copy into a .pl file and rename the predicate
        boolean_matrix_to_integers(b, "b", current_dir + "/output2.pl")

    def test_convert_lists_to_matrix(self):
        lists = [[0, 1], [0, 1]]
        a = lists_to_matrix(lists)

        self.assertEqual(a[0, 1], 1.0)
        self.assertEqual(a[1, 1], 1.0)
        self.assertNotEqual(a[0, 0], 1.0)
        self.assertNotEqual(a[1, 0], 1.0)


class TestBMLPTorchModules(unittest.TestCase):

    def test_simple_chain(self):
        # Create matrices for p1, p2, p3
        p1_indices = torch.tensor([[0], [1]], dtype=D_TYPE)
        p1_values = torch.tensor([1.0], dtype=D_TYPE)
        p1 = torch.sparse_coo_tensor(
            p1_indices, p1_values, (5, 5), device=GPU_DEVICE).to_dense()

        p2_indices = torch.tensor([[1], [2]], dtype=D_TYPE)
        p2_values = torch.tensor([1.0], dtype=D_TYPE)
        p2 = torch.sparse_coo_tensor(
            p2_indices, p2_values, (5, 5), device=GPU_DEVICE).to_dense()

        p3_indices = torch.tensor([[2], [3]], dtype=D_TYPE)
        p3_values = torch.tensor([1.0], dtype=D_TYPE)
        p3 = torch.sparse_coo_tensor(
            p3_indices, p3_values, (5, 5), device=GPU_DEVICE).to_dense()

        # Compute p0 = p1 * p2 * p3
        p0 = mul(mul(p1, p2), p3)

        self.assertTrue(p0[0, 3])
        self.assertTrue((p0[1:, :] == 0).all())

    def test_bmlp_rms_simple_recursion(self):

        # Create matrix R1
        num_nodes = 5
        indices = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        values = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=D_TYPE)
        R1 = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes), device=GPU_DEVICE).to_dense()

        res = RMS(R1)

        # Check all expected connections in the transitive closure
        expected_edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4),
            (3, 4)
        ]

        for src, dst in expected_edges:
            self.assertTrue(res[src, dst])

    def test_bmlp_rms_simple_recursion_with_same_body(self):

        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        num_nodes = 5

        # Create a square adjacency matrix using BOOL type
        R1 = new(num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = 1.0

        res = RMS(R1, R1)

        self.assertEqual(res[0, 1], 1.0)
        self.assertEqual(res[0, 2], 1.0)
        self.assertEqual(res[0, 3], 1.0)
        self.assertEqual(res[0, 4], 1.0)
        self.assertEqual(res[1, 2], 1.0)
        self.assertEqual(res[1, 3], 1.0)
        self.assertEqual(res[1, 4], 1.0)
        self.assertEqual(res[2, 3], 1.0)
        self.assertEqual(res[2, 4], 1.0)
        self.assertEqual(res[3, 4], 1.0)

    def test_bmlp_rms_simple_recursion_with_diff_body_1(self):

        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        num_nodes = 5

        # Create a square adjacency matrix using BOOL type
        R1 = new(num_nodes, num_nodes)
        R2 = new(num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges[1:]:
            R1[src, dst] = 1.0

        for src, dst in edges:
            R2[src, dst] = 1.0

        res = RMS(R1, R2)

        self.assertEqual(res[0, 2], 1.0)
        self.assertEqual(res[0, 3], 1.0)
        self.assertEqual(res[0, 4], 1.0)
        self.assertEqual(res[1, 2], 1.0)
        self.assertEqual(res[1, 3], 1.0)
        self.assertEqual(res[1, 4], 1.0)
        self.assertEqual(res[2, 3], 1.0)
        self.assertEqual(res[2, 4], 1.0)
        self.assertEqual(res[3, 4], 1.0)

    def test_bmlp_rms_simple_recursion_with_diff_body_2(self):

        # Create matrix R1
        p1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]])
        p2 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]])
        p3 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0, 0.0]])
        res = RMS(transpose(p1), mul(p2, p3))

        # Check all expected connections in the transitive closure
        self.assertTrue(res[0, 1], 1.0)
        self.assertTrue(res[1, 1], 1.0)
        self.assertTrue(res[3, 1], 1.0)

    def test_bmlp_smp_simple_recursion(self):
        num_nodes = 5

        # Create matrix R1
        indices = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 0, 4]])
        values = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=D_TYPE)
        R1 = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes), device=GPU_DEVICE).to_dense()

        # Create vector V (starting from node 3)
        V_indices = torch.tensor([[3]])
        V_values = torch.tensor([1.0], dtype=D_TYPE)
        V = torch.sparse_coo_tensor(
            V_indices, V_values, (num_nodes,), device=GPU_DEVICE).to_dense()

        res = SMP(V, R1)

        # Node 3 should reach nodes 0, 1, 2, 3 (not 4)
        res_dense = res
        for i in range(4):
            self.assertTrue(res_dense[i])
        self.assertFalse(res_dense[4])

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
            R1[src, dst] = 1.0

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = 1.0

        # Create a vector to represent a query
        V = new(2, num_cols)

        # query the reachability of node 1
        V[0, 0] = 1.0
        V[1, 1] = 1.0

        res, _ = IE(V, R1, R2)

        self.assertEqual(res[0, 0], 1.0)
        self.assertEqual(res[0, 1], 1.0)
        self.assertEqual(res[0, 2], 1.0)
        self.assertEqual(res[0, 3], 1.0)
        self.assertEqual(res[1, 1], 1.0)
        self.assertEqual(res[1, 2], 1.0)
        self.assertEqual(res[1, 3], 1.0)

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
            R1[src, dst] = 1.0

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = 1.0

        # Create a vector to represent a query
        V = new(2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = new(2, num_rows)

        # Query the reachability of node 0 and 3
        V[0, 0] = 1.0
        V[0, 2] = 1.0
        V[0, 3] = 1.0
        # Filter the second row in the matrix
        T[0, 0] = 1.0
        T[0, 2] = 1.0

        res, _ = IE(V, R1, R2, T)

        self.assertEqual(res[0, 0], 1.0)
        self.assertNotEqual(res[0, 1], 1.0)
        self.assertEqual(res[0, 2], 1.0)
        self.assertEqual(res[0, 3], 1.0)
        self.assertNotEqual(res[0, 4], 1.0)

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
            R1[src, dst] = 1.0

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = 1.0

        # Create a vector to represent a query
        V = new(2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = new(2, num_rows)

        # Query the reachability of node 0 and 3
        V[0, 0] = 1.0
        V[0, 3] = 1.0
        # Filter the second row in the matrix
        T[0, 1] = 1.0

        # Query the reachability of node 1
        V[1, 1] = 1.0
        # Filter the first row in the matrix
        T[1, 0] = 1.0

        res, _ = IE(V, R1, R2, T)

        self.assertEqual(res[0, 0], 1.0)
        self.assertEqual(res[0, 1], 1.0)
        self.assertNotEqual(res[0, 2], 1.0)
        self.assertEqual(res[0, 3], 1.0)
        self.assertNotEqual(res[0, 4], 1.0)

        self.assertNotEqual(res[1, 0], 1.0)
        self.assertEqual(res[1, 1], 1.0)
        self.assertEqual(res[1, 2], 1.0)
        self.assertEqual(res[1, 3], 1.0)
        self.assertEqual(res[1, 4], 1.0)

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
            R1[src, dst] = 1.0

        # Insert edges into the second matrix
        for src, dst in edges_2:
            R2[src, dst] = 1.0

        # Create a vector to represent a query
        V = new(2, num_cols)
        # Create a vector to represent a filter on the second matrix
        T = new(2, num_rows)

        # Node 0 should reach all other nodes
        V[0, 0] = 1.0

        # Filter the first row in the matrix
        T[1, 0] = 1.0
        T[1, 1] = 1.0
        T[1, 2] = 1.0
        # Now node 0 should not reach any other node
        V[1, 0] = 1.0

        res, _ = IE(V, R1, R2, T)

        self.assertEqual(res[0, 0], 1.0)
        self.assertEqual(res[0, 1], 1.0)
        self.assertEqual(res[0, 2], 1.0)
        self.assertEqual(res[0, 3], 1.0)
        self.assertEqual(res[0, 4], 1.0)

        self.assertEqual(res[1, 0], 1.0)
        self.assertNotEqual(res[1, 1], 1.0)
        self.assertNotEqual(res[1, 2], 1.0)
        self.assertNotEqual(res[1, 3], 1.0)
        self.assertNotEqual(res[1, 4], 1.0)


if __name__ == '__main__':
    unittest.main()
