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

import chex
import jax
import jax.numpy as jnp

from velora.compute.softmax import Softmax


def compute_importance_weights(
    pi_logits: chex.Array,
    mu_logits: chex.Array,
    actions: chex.Array,
) -> chex.Array:
    """
    Compute importance sampling weights for off-policy correction.

    Parameters
    ----------
    pi_logits : chex.Array
        Current policy logits. Shape: `(T, B, A)`
    mu_logits : chex.Array
        Behavior policy logits. Shape: `(T, B, A)`
    actions : chex.Array
        Actions taken. Shape: `(T, B)`

    Returns
    -------
    rho : chex.Array
        Importance weights. Shape: `(T, B)`
    """
    log_pi = Softmax(pi_logits).log_prob(actions)
    log_mu = Softmax(mu_logits).log_prob(actions)
    rho = jax.lax.stop_gradient(jnp.exp(log_pi - log_mu))  # type: ignore
    return rho


def transform_to_2hot(
    scalar: chex.Array,
    min_value: float,
    max_value: float,
    num_bins: int,
) -> chex.Array:
    """
    Encode scalars as 2-hot categorical distributions over a uniform support.

    Each scalar is clamped to `[min_value, max_value]`, mapped to a
    continuous bin index, and the probability mass is split linearly
    between the two nearest bins.

    Parameters
    ----------
    scalar : chex.Array
        Scalar values to encode.
    min_value : float
        Minimum representable value.
    max_value : float
        Maximum representable value.
    num_bins : int
        Number of discrete bins.

    Returns
    -------
    probs : chex.Array
        `(..., num_bins)` categorical distribution.
    """
    scalar = jnp.clip(scalar, min_value, max_value)

    # Continuous bin index in [0, num_bins - 1]
    bin_width = (max_value - min_value) / (num_bins - 1)
    index = (scalar - min_value) / bin_width

    lower = jnp.floor(index).astype(jnp.int32)
    upper_weight = index - lower  # fractional part
    lower_weight = 1.0 - upper_weight

    # Clamp upper bin index (when scalar == max_value, lower == num_bins - 1)
    upper = jnp.minimum(lower + 1, num_bins - 1)

    lower_one_hot = jax.nn.one_hot(lower, num_bins)
    upper_one_hot = jax.nn.one_hot(upper, num_bins)

    return (
        lower_one_hot * lower_weight[..., None]
        + upper_one_hot * upper_weight[..., None]
    )


def transform_from_2hot(
    probs: chex.Array,
    min_value: float,
    max_value: float,
    num_bins: int,
) -> chex.Array:
    """
    Decode a categorical distribution back to scalar (expected value).

    Parameters
    ----------
    probs : chex.Array
        Probability distribution over bins `(..., num_bins)`.
    min_value : float
        Minimum representable value.
    max_value : float
        Maximum representable value.
    num_bins : int
        Number of discrete bins.

    Returns
    -------
    scalar : chex.Array
        Expected scalar values.
    """
    support = jnp.linspace(min_value, max_value, num_bins)
    return jnp.sum(probs * support, axis=-1)
