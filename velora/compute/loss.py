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

from velora.compute.rl_ops import compute_gaussian_kl


def compute_gaussian_entropy_loss(
    log_std: jax.Array,
    coef: float = 1e-2,
    action_dim_mask: jax.Array | None = None,
) -> jax.Array:
    """
    Compute Gaussian entropy loss for policy regularization.

    Returns negative entropy so that minimizing the loss maximizes entropy.

    Parameters
    ----------
    log_std : jax.Array
        Policy log standard deviation. Shape: `(B, T, D)`
    coef : float (optional)
        Entropy coefficient. Default is `0.01`
    action_dim_mask : jax.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    loss : jax.Array
        Scalar negative mean entropy loss
    """
    # Per-dimension entropy: ½ + ½ log(2π) + log σ
    per_dim_entropy = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std

    if action_dim_mask is not None:
        per_dim_entropy = per_dim_entropy * action_dim_mask

    entropy = jnp.sum(per_dim_entropy, axis=-1)  # (B, T)
    return -coef * jnp.mean(entropy)


def compute_gaussian_policy_gradient_loss(
    mu: jax.Array,
    log_std: jax.Array,
    actions: jax.Array,
    advantages: jax.Array,
    action_dim_mask: jax.Array | None = None,
) -> jax.Array:
    """
    Compute differentiable policy gradient loss for continuous actions.

    Calculates `-log π(a|s) * A(s, a)` where `π` is a diagonal Gaussian.

    Parameters
    ----------
    mu : jax.Array
        Policy mean. Shape: `(B, T, D)`
    log_std : jax.Array
        Policy log standard deviation. Shape: `(B, T, D)`
    actions : jax.Array
        Actions taken. Shape: `(B, T, D)`
    advantages : jax.Array
        Advantage estimates. Shape: `(T, B)`
    action_dim_mask : jax.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    loss : jax.Array
        Per-timestep policy gradient loss. Shape: `(B, T-1)`
    """
    mu = mu[:, :-1]  # (B, T-1, D)
    log_std = log_std[:, :-1]  # (B, T-1, D)
    actions = actions[:, :-1]  # (B, T-1, D)

    advantages = jnp.transpose(advantages)  # (T-1, B) -> (B, T-1)

    # Gaussian log probability per dimension
    std = jnp.exp(log_std)
    per_dim_log_prob = (
        -0.5 * jnp.square((actions - mu) / (std + 1e-8))
        - log_std
        - 0.5 * jnp.log(2 * jnp.pi)
    )

    if action_dim_mask is not None:
        per_dim_log_prob = per_dim_log_prob * action_dim_mask

    log_prob = jnp.sum(per_dim_log_prob, axis=-1)  # (B, T-1)

    return -log_prob * jax.lax.stop_gradient(advantages)


def compute_gaussian_aux_policy_loss(
    aux_pi_pred: jax.Array,
    next_mu: jax.Array,
    next_log_std: jax.Array,
    discounts: jax.Array,
    action_dim_mask: jax.Array | None = None,
) -> jax.Array:
    """
    Compute auxiliary 1-step policy prediction loss for continuous actions.

    The auxiliary policy `p(s, a)` predicts the next-step Gaussian parameters
    `(μ', log σ')`, masked by episode boundaries.

    Unlike the discrete variant, no action indexing is needed — the prediction
    is already conditioned on the action taken via the ContinuousACM input.

    Parameters
    ----------
    aux_pi_pred : jax.Array
        Predicted next-step Gaussian parameters `(μ', log σ')`.
        Shape: `(B, T, 2D)` where `D = max_action_dim`
    next_mu : jax.Array
        Actual policy mean at next timestep. Shape: `(B, T, D)`
    next_log_std : jax.Array
        Actual policy log-std at next timestep. Shape: `(B, T, D)`
    discounts : jax.Array
        Episode continuation signals. Shape: `(B, T, 1)`
    action_dim_mask : jax.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    loss : jax.Array
        Per-timestep auxiliary policy loss. Shape: `(B, T-1)`
    """
    max_action_dim = (
        jnp.shape(aux_pi_pred)[-1] // 2
    )  # Static: 2D layout is (μ', log σ')

    # Split predicted (μ', log σ') from aux_pi output
    pred_mu = aux_pi_pred[:, :-1, :max_action_dim]  # (B, T-1, D)
    pred_log_std = aux_pi_pred[:, :-1, max_action_dim:]  # (B, T-1, D)

    # Actual next-step policy (stop gradient — this is the target)
    target_mu = jax.lax.stop_gradient(next_mu[:, 1:])  # (B, T-1, D)
    target_log_std = jax.lax.stop_gradient(next_log_std[:, 1:])  # (B, T-1, D)

    # Gaussian KL between predicted and actual next-step policy
    loss = compute_gaussian_kl(
        target_mu,
        target_log_std,
        pred_mu,
        pred_log_std,
        action_dim_mask=action_dim_mask,
    )

    # Mask out terminal states
    return loss * jnp.squeeze(discounts[:, :-1], axis=-1)  # (B, T-1)
