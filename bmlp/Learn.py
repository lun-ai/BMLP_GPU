from .Generator import *
from .Task import Task
import numpy as np


def learn_task(pos, neg, primitives, cached={}, max_iter=-1, num_hypo=1, unary_ops=DEFAULT_UNARY_OPS, binary_ops=DEFAULT_BINARY_OPS, verbose=False):

    # Create a new task with positive and negative examples
    new_task = Task(pos, neg)
    print(f'No. pos example: {new_task.get_pn()}')
    print(f'No. neg example: {new_task.get_nn()}')

    generator = Generator(primitives, unary_ops=unary_ops,
                          binary_ops=binary_ops)

    current_iter = 0
    learned_prog = []
    new_preds = primitives

    while current_iter < max_iter or max_iter < 0:

        success, learned_prog, rest = generator.invent_predicates_for_task(
            new_task, cached=cached, num_hypo=num_hypo)

        if verbose:
            print(
                f'Target: {success} | No. Programs in Iter {current_iter}: {len(rest)} | No. Redundant Programs: {generator.get_redundancy()}')

        # If success, means we found the right program(s)
        if success:
            print(f'Success: {num_hypo} program(s) learned!')
            break
        # If no more programs, break
        elif len(rest) == 0:
            print('Terminated: due to no more new programs!')
            break

        # Add new invented predicates as primitives
        new_preds = rest + new_preds
        generator.update_predicates(new_preds)
        current_iter += 1

    return learned_prog
