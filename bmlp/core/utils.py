from pygraphblas import Matrix, types, Vector, BOOL


# Parse a boolean matrix from Prolog version of BMLP which has integers as rows
def parse_prolog_binary_codes(path):

    codes = []
    max_length = 0

    with open(path, 'r') as prolog:

        num_facts = 0

        for row in prolog:
            if "%" not in row and row != "\n":

                # Get the integer representation of the bitcode
                code, length = integer_to_binary_code(
                    int(row.replace(" ", "").replace("\n", "").split(",")[1].strip(").")))
                codes.append(code)
                max_length = max(max_length, length)
                num_facts += 1

    return codes, num_facts, max_length,


def integer_to_binary_code(n):
    len = n.bit_length()
    return [n >> i & 1 for i in range(0, len)], len


# Write graphBLAS matrix as a set of integers in a prolog file
def boolean_matrix_to_integers(matrix: Matrix.sparse, name, path):

    with open(path, 'w') as prolog:
        for i in range(matrix.nrows):

            out = 0
            for j in range(matrix.ncols - 1, -1, -1):

                # sparse matrix elements can be empty bits
                element: BOOL = matrix.get(i, j)
                out = (out << 1) | element if element else (out << 1)

            prolog.write('%s(%s,%s).\n' % (name, i, out))


# From a path to a prolog file containing a boolean matrix
# convert it into a graphBLAS matrix for computation
def integers_to_boolean_matrix(path, is_squared=False):

    bitcodes, nrows, ncols = parse_prolog_binary_codes(path)

    if is_squared:
        dim = max(nrows, ncols)
        matrix = Matrix.sparse(BOOL, dim, dim)
    else:
        matrix = Matrix.sparse(BOOL, nrows, ncols)

    for row in range(len(bitcodes)):
        for col in range(len(bitcodes[row])):

            if bitcodes[row][col]:
                matrix[row, col] = True

    return matrix
