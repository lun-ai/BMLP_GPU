import pygraphblas as gb
import unittest
from matrix import *

class BMLPTests(unittest.TestCase):
    
    def test_load_pl(self):
        
        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix("bmlp/load_pl_test.pl")
        
        self.assertEqual(a[0,1],True)
        self.assertEqual(a[1,2],True)
        self.assertEqual(a[2,3],True)
        self.assertEqual(a[3,0],True)
        self.assertEqual(a[4,4],True)
        
    def test_save_pl(self):
        
        # Test Prolog to BMLP-GPU conversion
        # Load a .pl file containing rows of a boolean matrix
        a = integers_to_boolean_matrix("bmlp/load_pl_test.pl")
        
        # Create a copy of this boolean matrix
        b = a

        # Save this copy into a .pl file and rename the predicate
        boolean_matrix_to_integers(b, "b", "bmlp/output.pl")
    
    
    def test_simple_chain(self):
        
        # Create a sequence of boolean matrix operations to compute
        #   p0(X1, X2) :- p1(X1, X3), p2(X3, X4), p3(X4, X2).
        
        num_nodes = 5
        
        p1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        p2 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        p3 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        p1[0,1] = True
        p2[1,2] = True
        p3[2,3] = True

        # exactly-two-connected program in chained H2m
        p0 = p1 @ p2 @ p3
        
        self.assertEqual(p0[0,3],True)
        

    def test_bmlp_rms_simple_recursion(self):
        
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        num_nodes = 5
        
        # Create a square adjacency matrix using BOOL type
        R1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = True
        
        res = BMLP_RMS(R1)
        
        self.assertEqual(res[0,1],True) 
        self.assertEqual(res[0,2],True)
        self.assertEqual(res[0,3],True)
        self.assertEqual(res[0,4],True)
        self.assertEqual(res[1,2],True)
        self.assertEqual(res[1,3],True)
        self.assertEqual(res[1,4],True)
        self.assertEqual(res[2,3],True)
        self.assertEqual(res[2,4],True)
        self.assertEqual(res[3,4],True)
        
    def test_bmlp_rms_simple_recursion_with_same_body(self):
        
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        num_nodes = 5
        
        # Create a square adjacency matrix using BOOL type
        R1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        # Insert edges into the adjacency matrix
        for src, dst in edges:
            R1[src, dst] = True
        
        res = BMLP_RMS(R1,R1)
        
        self.assertEqual(res[0,1],True) 
        self.assertEqual(res[0,2],True)
        self.assertEqual(res[0,3],True)
        self.assertEqual(res[0,4],True)
        self.assertEqual(res[1,2],True)
        self.assertEqual(res[1,3],True)
        self.assertEqual(res[1,4],True)
        self.assertEqual(res[2,3],True)
        self.assertEqual(res[2,4],True)
        self.assertEqual(res[3,4],True)
    
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
        
        res = BMLP_RMS(R1,R2)
        
        self.assertEqual(res[0,2],True)
        self.assertEqual(res[0,3],True)
        self.assertEqual(res[0,4],True)
        self.assertEqual(res[1,2],True)
        self.assertEqual(res[1,3],True)
        self.assertEqual(res[1,4],True)
        self.assertEqual(res[2,3],True)
        self.assertEqual(res[2,4],True)
        self.assertEqual(res[3,4],True)
        
    def test_bmlp_rms_simple_recursion_with_diff_body_2(self):
        
        # Create a sequence of boolean matrix operations to compute
        #   p0(X1, X2) :- p1(X2, X1).
        #   p0(X1, X2) :- p2(X1, X3), p3(X3, X4), p0(X4, X2).
        num_nodes = 5
        p1 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        p2 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
        p3 = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)

        p1[1,0] = True

        p2[1,2] = True
        p2[3,4] = True

        p3[2,3] = True
        p3[4,0] = True

        # exactly-two-connected recursion in chained H2m
        res = BMLP_RMS(p1.T,p2 @ p3)

        self.assertEqual(res[0,1],True) 
        self.assertEqual(res[1,1],True) 
        self.assertEqual(res[3,1],True) 

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
        
        res = BMLP_SMP(V,R1)
        
        self.assertEqual(res[0],True) 
        self.assertEqual(res[1],True) 
        self.assertEqual(res[2],True) 
        self.assertEqual(res[3],True)
    

if __name__ == '__main__':
    unittest.main()