from .Generator import *
from .Task import Task
import cupy as cp

cp.random.seed(0)


def learn_task(pos, neg, primitives, cached={}, max_iter=-1, num_hypo=1, unary_ops=DEFAULT_UNARY_OPS, binary_ops=DEFAULT_BINARY_OPS, verbose=True):

    # Create a new task with positive and negative examples
    new_task = Task(pos, neg)
    if verbose:
        print(f'No. pos example: {new_task.get_pn()}')
        print(f'No. neg example: {new_task.get_nn()}')

    generator = Generator(primitives, unary_ops=unary_ops,
                          binary_ops=binary_ops)

    current_iter = 0
    learned_prog = []
    all_preds = primitives

    while current_iter < max_iter or max_iter < 0:

        # Generate new predicates
        success, learned_prog, new_preds = generator.invent_predicates_for_task(
            new_task, cached=cached, num_hypo=num_hypo)

        current_iter += 1
        if verbose:
            print(
                f'Iter {current_iter} - No. New Programs: {len(new_preds)} | No. Redundant Programs: {generator.get_redundancy()} | No. Eliminated Programs: {generator.get_elimination()}')

        # If success, means we found the right program(s)
        if success:
            if verbose:
                print(f'Success: {num_hypo} program(s) learned!')
            break
        # If no more programs, break
        elif len(new_preds) == 0:
            if verbose:
                print('Terminated: due to no more new programs!')
            break

        # Add new invented predicates as primitives
        all_preds = new_preds + all_preds
        generator.update_predicates(len(new_preds), all_preds)

    return learned_prog
