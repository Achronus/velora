# Copyright 2025 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import List

import chex
import jax
import jax.numpy as jnp


def to_time_first(x: jax.Array) -> jax.Array:
    """
    Transpose from batch-first to time-first format for scan operations.

    Parameters
    ----------
    x : jax.Array
        Input tensor in batch-first format `(B, T, ...)`

    Returns
    -------
    x : jax.Array
        Output tensor in time-first format `(T, B, ...)`
    """
    # (B, T, ...) -> (T, B, ...)
    return jnp.swapaxes(x, 0, 1)


def to_batch_first(x: jax.Array) -> jax.Array:
    """
    Transpose from time-first to batch-first format after scan operations.

    Parameters
    ----------
    x : jax.Array
        Input tensor in time-first format `(T, B, ...)`

    Returns
    -------
    x : jax.Array
        Output tensor in batch-first format `(B, T, ...)`
    """
    # (T, B, ...) -> (B, T, ...)
    return jnp.swapaxes(x, 0, 1)


def squeeze_time(x: jax.Array) -> jax.Array:
    """
    Squeeze the time dimension (axis 1) if `T=1`.

    Useful for removing redundant sequence dimensions when
    processing single timestep observations.

    Parameters
    ----------
    x : jax.Array
        Input tensor with shape `(B, T, ...)` where T may be 1

    Returns
    -------
    x : jax.Array
        Output tensor with shape `(B, ...)` if `T=1`, otherwise unchanged
    """
    if x.ndim >= 2 and jnp.shape(x)[1] == 1:
        return jnp.squeeze(x, axis=1)
    return x


def stack_pytrees(items: List) -> chex.ArrayTree:
    """
    Stack a list of pytrees along a new leading axis.

    Acts as a thin wrapper around `jax.tree.map` + `jnp.stack`.
    Each element of `items` must be a pytree with identical structure and
    leaf shapes.

    Parameters
    ----------
    items : List[chex.ArrayTree]
        List of pytrees to stack. All elements must share the same
        structure and leaf shapes

    Returns
    -------
    stacked : chex.ArrayTree
        A single pytree where every leaf has shape `(len(items), ...)`
    """
    return jax.tree.map(lambda *xs: jnp.stack(xs), *items)


def unstack_pytree(tree: chex.ArrayTree, i: int) -> chex.ArrayTree:
    """
    Extract the `i`-th element from every leaf of a pytree.

    The inverse of `stack_pytrees` — where `stack_pytrees` combines a
    list of pytrees into a single batched pytree with a leading group
    dimension, `unstack_pytrees` slices out one element along that
    dimension.

    Parameters
    ----------
    tree : chex.ArrayTree
        A pytree whose leaves have a leading batch dimension, typically
        the output of a vmapped function
    i : int
        Index to extract along the leading dimension of every leaf

    Returns
    -------
    sliced : chex.ArrayTree
        A pytree with the same structure as `tree` but with every leaf
        reduced from shape `(B, ...)` to `(...,)`
    """
    return jax.tree.map(lambda x: x[i], tree)
