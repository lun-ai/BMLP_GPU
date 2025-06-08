import torch
from bmlp.core.tensor import *
from bmlp.core.utils import *

# Extract relations from a Prolog file and create matrices
unary, binary = extract_relations_from_file('bmlp/tests/ex_p0.pl')
data = create_matrices_from_relations('edge',
                                      ['node', 'node'],
                                      unary, binary)

# Convert the data to a PyTorch tensor and apply RMS
m1 = torch.tensor(data['matrix'],
                  dtype=D_TYPE)
m2 = RMS(m1)

# Print the result
print('RMS result:\n', m2)
print_relations(convert_matrix_to_relations(
    m2, data['index_to_entity']['node']))
