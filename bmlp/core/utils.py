import pickle
import os
import re
import numpy as np
import pandas as pd
from graphblas import Matrix
from graphblas.dtypes import *
import torch


# Parse a boolean matrix from Prolog version of BMLP which has integers as rows
# return a list of bitcodes arrays, the number of facts and the maximum length of the bitcodes
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


def boolean_matrix_to_integers(matrix: Matrix, name='abdm', path='cm.pl'):

    cm = []
    is_using_tensor = torch.cuda.is_available() and isinstance(matrix, torch.Tensor)

    if is_using_tensor:
        nrows = matrix.shape[0]
        ncols = matrix.shape[1]
    else:
        nrows = matrix.nrows
        ncols = matrix.ncols

    with open(path, 'w') as prolog:
        for i in range(nrows):

            out = 0
            for j in range(ncols - 1, -1, -1):

                # sparse matrix elements can be empty bits
                if is_using_tensor:
                    element = matrix[i, j].item()
                    out = (out << 1) | (element > 0.0)
                else:
                    element = matrix.get(i, j)
                    # element may be None
                    out = (out << 1) | element if element else (out << 1)

            prolog.write('%s(%s,%s).\n' % (name, i, out))
            cm.append(out)
    return cm


# From a path to a prolog file containing a boolean matrix
# convert it into a graphBLAS matrix for computation
def integers_to_boolean_matrix(path, is_squared=False, to_tensor=False):

    is_using_tensor = torch.cuda.is_available() and to_tensor
    bitcodes, nrows, ncols = parse_prolog_binary_codes(path)

    # If the matrix needs to be squared, create a square matrix
    if is_squared:
        dim = max(nrows, ncols)
        if is_using_tensor:
            matrix = torch.zeros(dim, dim)
        else:
            matrix = Matrix(BOOL, dim, dim)
    else:
        if is_using_tensor:
            matrix = torch.zeros(nrows, ncols)
        else:
            matrix = Matrix(BOOL, nrows, ncols)

    for row in range(len(bitcodes)):
        for col in range(len(bitcodes[row])):

            if bitcodes[row][col]:
                if is_using_tensor:
                    matrix[row, col] = 1.0
                else:
                    matrix[row, col] = True

    return matrix


# From a python list of lists to a graphBLAS matrix
def lists_to_matrix(lists: list[list[int]]):
    matrix = Matrix.from_dense(lists, missing_value=0)
    return matrix


def extract_relations_from_file(file_path: str) -> tuple[list[tuple], list[tuple]]:
    """
    Extracts unary and binary relations from a given file.
    Returns two lists: unary relations and binary relations.
    """
    unary_relations = []
    binary_relations = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.replace(' ', '')
            # Match unary relations
            unary_match = re.match(r'(\w+)\((\w+)\).', line)
            if unary_match:
                relation_type, entity = unary_match.groups()
                unary_relations.append((relation_type, entity))

            # Match binary relations
            binary_match = re.match(r'(\w+)\((\w+),(\w+)\).', line)
            if binary_match:
                relation, entity1, entity2 = binary_match.groups()
                binary_relations.append((relation, entity1, entity2))

    return unary_relations, binary_relations


def convert_matrix_to_relations(matrix: np.ndarray, index_to_entity: dict) -> list[tuple]:
    """
    Converts a numpy matrix to a list of binary relations.
    Each relation is represented as a tuple (relation, entity1, entity2).
    """
    relations = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                entity1 = index_to_entity[i]
                entity2 = index_to_entity[j]
                relations.append((f'relation', entity1, entity2))
    return relations


def print_relations(binary_relations):
    """
    Prints binary relations in a readable format.
    """
    for relation in binary_relations:
        print(f"{relation[0]}({relation[1]}, {relation[2]})")


def save_relations_to_csv(unary_relations, binary_relations, dir_path='', verbose=False):
    """
    Saves unary and binary relations to CSV files in the specified directory.
    """
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Save unary relations
    unary_df = pd.DataFrame(unary_relations, columns=['type', 'entity'])
    unary_df.to_csv(os.path.join(dir_path, 'unary_relations.csv'), index=False)

    # Save binary relations
    binary_df = pd.DataFrame(binary_relations, columns=[
                             'relation', 'entity1', 'entity2'])
    binary_df.to_csv(os.path.join(
        dir_path, 'binary_relations.csv'), index=False)

    if verbose:
        print(
            f"Number of entities in unary relations: {len(unary_df['entity'].unique())}")
        print(f"Number of binary relations: {len(binary_df)}")
        print(f"Saved unary relations to {dir_path}/unary_relations.csv")
        print(f"Saved binary relations to {dir_path}/binary_relations.csv")


def create_matrices_from_relations(predicate: str,
                                   types: list[str],
                                   unary_relations: list[tuple],
                                   binary_relations: list[tuple],
                                   verbose: bool = False) -> dict:
    """
    Extracts relations from a given file and returns a list of tuples.
    Each tuple contains the relation type and the entities involved.
    """
    # Find in binary relations all tuples that match the predicate
    filtered_relations = [
        rel for rel in binary_relations if rel[0] == predicate]

    # Find entities of the specified types in unary relations
    type_entities = {type_:
                     set([type_entity[1] for type_entity in unary_relations if type_ == type_entity[0]]) for type_ in types}
    # Add indexes to the entities for every type
    entity_to_index = {type_: {entity: idx for idx, entity in enumerate(sorted(entities))}
                       for type_, entities in type_entities.items()}
    index_to_entity = {type_: {idx: entity for entity, idx in entities.items()}
                       for type_, entities in entity_to_index.items()}

    # Initialize a matrix for the current type
    matrix_size = tuple([len(entity_to_index[type_]) for type_ in types])
    relation_matrix = np.zeros(matrix_size)

    # Fill the matrix with relations
    for rel in filtered_relations:
        entity_ids = tuple()
        for i, type_ in enumerate(types):
            entity = rel[1 + i]
            if entity in entity_to_index[type_]:
                idx = entity_to_index[type_][entity]
                entity_ids += (idx,)
        # Use the entity IDs to set the corresponding positions in the matrix
        # Ensure entity_ids has the correct number of elements before indexing
        assert len(entity_ids) == len(types)
        relation_matrix[entity_ids] = 1

        # Store the matrix in the relation_matrices dictionary
    relation = {
        'matrix': relation_matrix,
        'entity_to_index': entity_to_index,
        'index_to_entity': index_to_entity
    }

    if verbose:
        print(f"Matrix size for types {types}: {matrix_size}")
        print(
            f"Creating relation matrices for predicate '{predicate}' with types {types}...")
        print(
            f"Found {len(filtered_relations)} relations matching the predicate.")

    return relation
