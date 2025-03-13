import unittest
import torch
import numpy as np
import os
from bmlp.core.tensor import *


# Get absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get directory containing the current file
current_dir = os.path.dirname(current_file_path)


class TestCoreTorchMatrix(unittest.TestCase):

    def test_mul(self):
        # Create a square adjacency matrix
        a = torch.tensor(
            [[False, True, False],
             [False, False, True],
             [False, False, False]]
        )
        b = torch.tensor(
            [[False, True, False],
             [False, False, True],
             [False, False, False]]
        )

        c = mul(a, b)

        # Test with to_dense() for easier comparison
        self.assertTrue(c.to_dense()[0][2].item())
        self.assertFalse(c.to_dense()[1][2].item())
        self.assertFalse(c.to_dense()[0][1].item())

    def test_square(self):
        # Create a square adjacency matrix
        a = torch.tensor(
            [[False, True, False],
             [False, False, True],
             [False, False, False]]
        )

        b = square(a)

        self.assertTrue(b.to_dense()[0][2].item())
        self.assertFalse(b.to_dense()[1][2].item())
        self.assertFalse(b.to_dense()[0][1].item())

    def test_add(self):
        # Create square adjacency matrices
        a = torch.tensor(
            [[False, True, False],
             [False, False, True],
             [False, False, False]]
        )
        b = torch.tensor(
            [[False, True, True],
             [False, False, True],
             [False, False, False]]
        )

        c = add(a, b)

        self.assertTrue(c.to_dense()[0][1].item())
        self.assertTrue(c.to_dense()[0][2].item())
        self.assertTrue(c.to_dense()[1][2].item())
        self.assertFalse(c.to_dense()[0][0].item())
        self.assertFalse(c.to_dense()[1][0].item())
        self.assertFalse(c.to_dense()[1][1].item())

    def test_intersection(self):
        # Create square adjacency matrices
        a = torch.tensor(
            [[False, True, False],
             [False, False, True],
             [False, False, False]]
        )
        b = torch.tensor(
            [[False, True, True],
             [False, False, True],
             [False, False, False]]
        )

        c = intersection(a, b)

        self.assertTrue(c.to_dense()[0, 1])
        self.assertTrue(c.to_dense()[1, 2])
        self.assertFalse(c.to_dense()[0, 2])

    def test_transpose(self):
        # Create a square adjacency matrix
        a_indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        a_values = torch.tensor([True, True], dtype=torch.bool)
        a = torch.sparse_coo_tensor(a_indices, a_values, (3, 3))

        b = transpose(a)

        self.assertTrue(b.to_dense()[1][0].item())
        self.assertTrue(b.to_dense()[2][1].item())
        self.assertFalse(b.to_dense()[0][1].item())
        self.assertFalse(b.to_dense()[1][2].item())

    def test_negate(self):
        # Create a square adjacency matrix
        a = torch.tensor(
            [[False, True, False],
             [False, False, True],
             [False, False, False]]
        )

        b = negate(a)

        self.assertFalse(b.to_dense()[0][1].item())
        self.assertFalse(b.to_dense()[1][2].item())
        self.assertTrue(b.to_dense()[0][0].item())
        self.assertTrue(b.to_dense()[0][2].item())
        self.assertTrue(b.to_dense()[1][0].item())
        self.assertTrue(b.to_dense()[1][1].item())
        self.assertTrue(b.to_dense()[2][0].item())
        self.assertTrue(b.to_dense()[2][1].item())
        self.assertTrue(b.to_dense()[2][2].item())

    def test_identity(self):
        b = identity(2)

        self.assertTrue(b.to_dense()[0][0].item())
        self.assertTrue(b.to_dense()[1][1].item())
        self.assertFalse(b.to_dense()[0][1].item())
        self.assertFalse(b.to_dense()[1][0].item())

    def test_resize(self):
        # Create a square adjacency matrix
        a = torch.tensor(
            [[False, True, False],
             [False, False, True],
             [False, False, False]]
        )

        b = resize(a, 6, 6)
        b[5][5] = True

        self.assertTrue(b.to_dense()[0][1].item())
        self.assertTrue(b.to_dense()[1][2].item())
        self.assertFalse(b.to_dense()[0][0].item())
        self.assertFalse(b.to_dense()[0][2].item())
        self.assertFalse(b.to_dense()[1][0].item())
        self.assertFalse(b.to_dense()[1][1].item())
        self.assertFalse(b.to_dense()[2][3].item())
        self.assertFalse(b.to_dense()[3][0].item())
        self.assertFalse(b.to_dense()[4][4].item())
        self.assertTrue(b.to_dense()[5][5].item())


class TestIOTorch(unittest.TestCase):

    def test_save_load_matrix(self):
        # Create a square adjacency matrix
        a = torch.tensor([[False, True, False],
                          [False, False, True],
                          [False, False, False]])

        # Test save functionality
        save(a, current_dir + "/output_torch.csv")

        # Test load functionality
        b = load(current_dir + "/output_torch.csv")

        self.assertTrue(b.to_dense()[0][1].item())
        self.assertTrue(b.to_dense()[1][2].item())
        self.assertFalse(b.to_dense()[0][0].item())
        self.assertFalse(b.to_dense()[0][2].item())


# class TestBMLPTorchModules(unittest.TestCase):

    # def test_simple_chain(self):
    #     # Create matrices for p1, p2, p3
    #     p1_indices = torch.tensor([[0, 1]], dtype=torch.long)
    #     p1_values = torch.tensor([True], dtype=torch.bool)
    #     p1 = torch.sparse_coo_tensor(p1_indices, p1_values, (5, 5))

    #     p2_indices = torch.tensor([[1, 2]], dtype=torch.long)
    #     p2_values = torch.tensor([True], dtype=torch.bool)
    #     p2 = torch.sparse_coo_tensor(p2_indices, p2_values, (5, 5))

    #     p3_indices = torch.tensor([[2, 3]], dtype=torch.long)
    #     p3_values = torch.tensor([True], dtype=torch.bool)
    #     p3 = torch.sparse_coo_tensor(p3_indices, p3_values, (5, 5))

    #     # Compute p0 = p1 * p2 * p3
    #     p0 = mul(mul(p1, p2), p3)

    #     self.assertTrue(p0.to_dense()[0, 3])
    #     self.assertFalse((p0.to_dense()[1:, :] != 0).any())

    # def test_bmlp_rms_simple_recursion(self):
    #     edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    #     num_nodes = 5

    #     # Create matrix R1
    #     indices = torch.tensor([[src, dst] for src, dst in edges]).t()
    #     values = torch.ones(len(edges), dtype=torch.bool)
    #     R1 = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    #     res = RMS(R1)

    #     # Check all expected connections in the transitive closure
    #     expected_edges = [
    #         (0, 1), (0, 2), (0, 3), (0, 4),
    #         (1, 2), (1, 3), (1, 4),
    #         (2, 3), (2, 4),
    #         (3, 4)
    #     ]

    #     for src, dst in expected_edges:
    #         self.assertTrue(res.to_dense()[src, dst])

    # def test_bmlp_smp_simple_recursion(self):
    #     edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 4)]
    #     num_nodes = 5

    #     # Create matrix R1
    #     indices = torch.tensor([[src, dst] for src, dst in edges]).t()
    #     values = torch.ones(len(edges), dtype=torch.bool)
    #     R1 = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    #     # Create vector V (starting from node 3)
    #     V_indices = torch.tensor([[3]], dtype=torch.long)
    #     V_values = torch.tensor([True], dtype=torch.bool)
    #     V = torch.sparse_coo_tensor(V_indices, V_values, (num_nodes,))

    #     res = SMP(V, R1)

    #     # Node 3 should reach nodes 0, 1, 2, 3 (not 4)
    #     res_dense = res.to_dense()
    #     for i in range(4):
    #         self.assertTrue(res_dense[i])
    #     self.assertFalse(res_dense[4])


if __name__ == '__main__':
    unittest.main()
