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

import jax
import jax.numpy as jnp


def compute_l2_mean_penalty(x: jax.Array) -> jax.Array:
    """
    Compute L2 penalty on the mean of a tensor.

    Penalizes the global mean drifting from zero, encouraging
    centered outputs. Useful for regularizing meta-network targets
    to prevent divergence.

    Parameters
    ----------
    x : jax.Array
        Input tensor of any shape

    Returns
    -------
    penalty : jax.Array
        Scalar L2 penalty on the mean
    """
    return jnp.square(jnp.mean(x))


def categorical_kl_divergence(
    p_logits: jax.Array,
    q_logits: jax.Array,
) -> jax.Array:
    """
    KL divergence `KL(p || q)` between two categorical distributions.

    Both inputs are unnormalized logits; softmax is applied internally.

    Parameters
    ----------
    p_logits : jax.Array
        Logits of distribution `p`.
    q_logits : jax.Array
        Logits of distribution `q`.

    Returns
    -------
    kl : jax.Array
        Per-element KL divergence (summed over the category axis).
    """
    p_log = jax.nn.log_softmax(p_logits, axis=-1)
    q_log = jax.nn.log_softmax(q_logits, axis=-1)
    p = jax.nn.softmax(p_logits, axis=-1)
    return jnp.sum(p * (p_log - q_log), axis=-1)


def batched_index(
    values: jax.Array,
    indices: jax.Array,
) -> jax.Array:
    """
    Index into the last axis of `values` using integer `indices`.

    Equivalent to `values[..., indices]` but works cleanly with
    arbitrary leading batch dimensions via one-hot contraction.

    Parameters
    ----------
    values : jax.Array
        Values to index. Shape: `(..., A)`
    indices : jax.Array
        Integer indices. Shape: `(...)`

    Returns
    -------
    indexed : jax.Array
        Selected values. Shape: `(...)`
    """
    one_hot = jax.nn.one_hot(indices, jnp.shape(values)[-1])
    return jnp.sum(values * one_hot, axis=-1)
