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

from velora.compute.softmax import Softmax


def compute_softmax_importance_weights(
    pi_logits: jax.Array,
    mu_logits: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """
    Compute importance sampling weights for off-policy correction.

    Parameters
    ----------
    pi_logits : jax.Array
        Current policy logits. Shape: `(T, B, A)`
    mu_logits : jax.Array
        Behavior policy logits. Shape: `(T, B, A)`
    actions : jax.Array
        Actions taken. Shape: `(T, B)`

    Returns
    -------
    rho : jax.Array
        Importance weights. Shape: `(T, B)`
    """
    log_pi = Softmax(pi_logits).log_prob(actions)
    log_mu = Softmax(mu_logits).log_prob(actions)
    rho = jax.lax.stop_gradient(jnp.exp(log_pi - log_mu))
    return rho


def compute_gaussian_importance_weights(
    mu_pi: jax.Array,
    log_std_pi: jax.Array,
    mu_mu: jax.Array,
    log_std_mu: jax.Array,
    actions: jax.Array,
    action_dim_mask: jax.Array | None = None,
) -> jax.Array:
    """
    Compute importance sampling weights for off-policy correction
    with Gaussian policies.

    Parameters
    ----------
    mu_pi : jax.Array
        Current policy mean. Shape: `(T, B, D)`
    log_std_pi : jax.Array
        Current policy log-std. Shape: `(T, B, D)`
    mu_mu : jax.Array
        Behavior policy mean. Shape: `(T, B, D)`
    log_std_mu : jax.Array
        Behavior policy log-std. Shape: `(T, B, D)`
    actions : jax.Array
        Actions taken. Shape: `(T, B, D)`
    action_dim_mask : jax.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    rho : jax.Array
        Importance weights. Shape: `(T, B)`
    """

    def _log_prob(mu, log_std, a):
        std = jnp.exp(log_std)
        per_dim = (
            -0.5 * jnp.square((a - mu) / (std + 1e-8))
            - log_std
            - 0.5 * jnp.log(2 * jnp.pi)
        )
        if action_dim_mask is not None:
            per_dim = per_dim * action_dim_mask

        return jnp.sum(per_dim, axis=-1)

    log_pi = _log_prob(mu_pi, log_std_pi, actions)
    log_mu = _log_prob(mu_mu, log_std_mu, actions)
    rho = jax.lax.stop_gradient(jnp.exp(log_pi - log_mu))
    return rho


def transform_to_2hot(
    scalar: jax.Array,
    min_value: float,
    max_value: float,
    num_bins: int,
) -> jax.Array:
    """
    Encode scalars as 2-hot categorical distributions over a uniform support.

    Each scalar is clamped to `[min_value, max_value]`, mapped to a
    continuous bin index, and the probability mass is split linearly
    between the two nearest bins.

    Parameters
    ----------
    scalar : jax.Array
        Scalar values to encode.
    min_value : float
        Minimum representable value.
    max_value : float
        Maximum representable value.
    num_bins : int
        Number of discrete bins.

    Returns
    -------
    probs : jax.Array
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
    probs: jax.Array,
    min_value: float,
    max_value: float,
    num_bins: int,
) -> jax.Array:
    """
    Decode a categorical distribution back to scalar (expected value).

    Parameters
    ----------
    probs : jax.Array
        Probability distribution over bins `(..., num_bins)`.
    min_value : float
        Minimum representable value.
    max_value : float
        Maximum representable value.
    num_bins : int
        Number of discrete bins.

    Returns
    -------
    scalar : jax.Array
        Expected scalar values.
    """
    support = jnp.linspace(min_value, max_value, num_bins)
    return jnp.sum(probs * support, axis=-1)


def compute_gaussian_kl(
    mu_target: jax.Array,
    log_std_target: jax.Array,
    mu_pred: jax.Array,
    log_std_pred: jax.Array,
    action_dim_mask: jax.Array | None = None,
) -> jax.Array:
    """
    Compute KL divergence between two diagonal Gaussian distributions.

    `KL(target || pred)` computed analytically per dimension, then summed
    over valid action dimensions.

    Parameters
    ----------
    mu_target : jax.Array
        Target mean. Shape: `(B, T, D)`
    log_std_target : jax.Array
        Target log standard deviation. Shape: `(B, T, D)`
    mu_pred : jax.Array
        Predicted mean. Shape: `(B, T, D)`
    log_std_pred : jax.Array
        Predicted log standard deviation. Shape: `(B, T, D)`
    action_dim_mask : jax.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    kl : jax.Array
        Per-sample KL divergence. Shape: `(B, T)`
    """
    std_target = jnp.exp(log_std_target)
    std_pred = jnp.exp(log_std_pred)
    var_pred = jnp.square(std_pred)

    # Per-dimension KL: log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²) / (2σ₂²) - ½
    per_dim_kl = (
        log_std_pred
        - log_std_target
        + (jnp.square(std_target) + jnp.square(mu_target - mu_pred))
        / (2.0 * var_pred + 1e-8)
        - 0.5
    )

    if action_dim_mask is not None:
        per_dim_kl = per_dim_kl * action_dim_mask

    return jnp.sum(per_dim_kl, axis=-1)  # (B, T)
