import cupy as cp
import numpy as np
from bmlp.core import matrix
import random
import time

# Complex graph
num_nodes = 20000
p = 0.0001

num_reps = 10
total_time = []

# repeat num_reps times
for i in range(num_reps):

    # Create a square adjacency matrix using BOOL type
    empty_matrix = matrix.new(num_nodes, num_nodes)
    R1 = empty_matrix

    # sample edges with edge probability < p
    for i in range(num_nodes):
        for j in range(num_nodes):
            if random.random() < p:
                R1[i, j] = True

    print('Matrix created')

    # R1.to_binfile("large")

    # Run and time the BMLP-RMS module
    start_time = time.time()
    matrix.RMS(R1)
    end_time = time.time()

    total_time.append(end_time - start_time)
    print('Wall time: ' + str(end_time - start_time))

print('Mean wall time: ' + str(np.mean(total_time)))
print('Wall time sterr: ' + str(np.std(total_time) / np.sqrt(len(total_time))))
