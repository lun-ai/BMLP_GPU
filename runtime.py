import random, time
import pygraphblas as gb

# An even larer graph
num_nodes = 5000
all_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes)]
print('No. edges: ' + str(len(all_edges)))

num_reps = 10
p = 0.5
total_time = 0

# repeat num_reps times
for i in range(num_reps):

    # sample edges with edge probability < p
    sampled_edges = list(filter(lambda _: random.random() < p, all_edges))

    # Create a square adjacency matrix using BOOL type
    empty_matrix = gb.Matrix.sparse(gb.BOOL, num_nodes, num_nodes)
    R1 = empty_matrix

    # Insert edges into the adjacency matrix
    for src, dst in sampled_edges:
        R1[src, dst] = True

    # Run and time the BMLP-RMS module
    start_time = time.time()
    R2 = R1 @ R1
    end_time = time.time()

    total_time += end_time - start_time
    print('Wall time: ' + str(end_time - start_time))

print('Mean wall time: ' + str(total_time / num_reps))

# Wall time: 1.3829445838928223
# Wall time: 1.3503551483154297
# Wall time: 1.3496010303497314
# Wall time: 1.3444621562957764
# Wall time: 1.3522448539733887
# Wall time: 1.3409295082092285
# Wall time: 1.3878204822540283
# Wall time: 1.3518271446228027
# Wall time: 1.410310983657837
# Wall time: 1.3928775787353516
# Mean wall time: 1.3663373470306397