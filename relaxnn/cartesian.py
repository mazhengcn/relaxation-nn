import torch
from torch.func import vmap


def cartesian_prod(*tensors):
    """
    This function is used to generate the cartesian product of a list of tensors.

    Args:
        tensors: Rank 2 tensor with shape (N_i, d).

    Returns:
        Cartesian product of the input tensors with shape (N_1, N_2, ..., N_n, nd).
    """

    num_tensors = len(tensors)

    def concat_fn(tensors):
        return torch.cat(tensors, dim=-1)

    for i in range(num_tensors):
        in_axes = [None] * num_tensors
        in_axes[-i - 1] = int(0)
        concat_fn = vmap(concat_fn, in_dims=(tuple(in_axes),))

    return concat_fn(tensors)
