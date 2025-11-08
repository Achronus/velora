import jax

import numpy as np
import flax.nnx as nnx


def total_parameters(model: nnx.Module) -> int:
    """
    Calculates the total number of parameters used in a Flax `nnx.Module`.

    Parameters:
        model (nnx.Module): a Flax module with parameters

    Returns:
        count (int): the total number of parameters.
    """
    params = nnx.state(model, nnx.Param)
    return np.sum([np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)])


def active_parameters(model: nnx.Module) -> int:
    """
    Calculates the active number of parameters used in a Flax `nnx.Module`.
    Filters out parameters that are `0`.

    Parameters:
        model (nnx.Module): a Flax module with parameters

    Returns:
        count (int): the total active number of parameters.
    """
    params = nnx.state(model, nnx.Param)
    return np.sum([p[(p != 0)].shape for p in jax.tree_util.tree_leaves(params)])
