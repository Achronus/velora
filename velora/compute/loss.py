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

from velora.compute.rl_ops import compute_gaussian_kl
from velora.compute.softmax import Softmax
from velora.compute.utils import batched_index


def compute_entropy_loss(logits: chex.Array, coef: float = 1e-2) -> chex.Array:
    """
    Compute entropy loss for policy regularization.

    Encourages exploration by penalizing low-entropy (overly deterministic)
    policies. Returns negative entropy so that minimizing the loss
    maximizes entropy.

    Parameters
    ----------
    logits : chex.Array
        Policy logits. Shape: `(B, T, A)` or `(B, A)`

        - batch_size (`B`) - the number of samples per timestep
        - seq_length (`T`) - the number of timesteps (optional)
        - n_actions (`A`) - the number of discrete actions

    coef : float (optional)
        Entropy coefficient. Higher values promote more exploration.
        Default is `0.01`

    Returns
    -------
    loss : chex.Array
        Scalar negative mean entropy loss
    """
    return -coef * jnp.mean(Softmax(logits).entropy())


def compute_discrete_policy_gradient_loss(
    logits: chex.Array,
    actions: chex.Array,
    advantages: chex.Array,
) -> chex.Array:
    """
    Compute differentiable policy gradient loss (REINFORCE-style).

    Calculates `-log π(a|s) * A(s, a)` for each timestep. Advantages are
    stopped from gradient flow as they should not influence the meta-network
    through this path.

    Parameters
    ----------
    logits : chex.Array
        Policy logits. Shape: `(B, T, A)`
    actions : chex.Array
        Actions taken. Shape: `(B, T, 1)`
    advantages : chex.Array
        Advantage estimates. Shape: `(T, B)`

    Returns
    -------
    loss : chex.Array
        Per-timestep policy gradient loss. Shape: `(B, T)`
    """
    logits = logits[:, :-1]  # (B, T-1), # type: ignore
    actions = actions[:, :-1]  # (B, T-1, 1), # type: ignore

    advantages = jnp.transpose(advantages)  # (B, T)
    actions = jnp.squeeze(actions, axis=-1)  # (B, T)

    log_pi = jax.nn.log_softmax(logits)
    log_pi_a = batched_index(log_pi, actions)
    return -log_pi_a * jax.lax.stop_gradient(advantages)  # type: ignore


def compute_gaussian_entropy_loss(
    log_std: chex.Array,
    coef: float = 1e-2,
    action_dim_mask: chex.Array | None = None,
) -> chex.Array:
    """
    Compute Gaussian entropy loss for policy regularization.

    Returns negative entropy so that minimizing the loss maximizes entropy.

    Parameters
    ----------
    log_std : chex.Array
        Policy log standard deviation. Shape: `(B, T, D)`
    coef : float (optional)
        Entropy coefficient. Default is `0.01`
    action_dim_mask : chex.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    loss : chex.Array
        Scalar negative mean entropy loss
    """
    # Per-dimension entropy: ½ + ½ log(2π) + log σ
    per_dim_entropy = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std

    if action_dim_mask is not None:
        per_dim_entropy = per_dim_entropy * action_dim_mask

    entropy = jnp.sum(per_dim_entropy, axis=-1)  # (B, T)
    return -coef * jnp.mean(entropy)


def compute_gaussian_policy_gradient_loss(
    mu: chex.Array,
    log_std: chex.Array,
    actions: chex.Array,
    advantages: chex.Array,
    action_dim_mask: chex.Array | None = None,
) -> chex.Array:
    """
    Compute differentiable policy gradient loss for continuous actions.

    Calculates `-log π(a|s) * A(s, a)` where `π` is a diagonal Gaussian.

    Parameters
    ----------
    mu : chex.Array
        Policy mean. Shape: `(B, T, D)`
    log_std : chex.Array
        Policy log standard deviation. Shape: `(B, T, D)`
    actions : chex.Array
        Actions taken. Shape: `(B, T, D)`
    advantages : chex.Array
        Advantage estimates. Shape: `(T, B)`
    action_dim_mask : chex.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    loss : chex.Array
        Per-timestep policy gradient loss. Shape: `(B, T-1)`
    """
    mu = mu[:, :-1]  # (B, T-1, D), # type: ignore
    log_std = log_std[:, :-1]  # (B, T-1, D), # type: ignore
    actions = actions[:, :-1]  # (B, T-1, D), # type: ignore

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
    aux_pi_pred: chex.Array,
    next_mu: chex.Array,
    next_log_std: chex.Array,
    discounts: chex.Array,
    action_dim_mask: chex.Array | None = None,
) -> chex.Array:
    """
    Compute auxiliary 1-step policy prediction loss for continuous actions.

    The auxiliary policy `p(s, a)` predicts the next-step Gaussian parameters
    `(μ', log σ')`, masked by episode boundaries.

    Unlike the discrete variant, no action indexing is needed — the prediction
    is already conditioned on the action taken via the ContinuousACM input.

    Parameters
    ----------
    aux_pi_pred : chex.Array
        Predicted next-step Gaussian parameters `(μ', log σ')`.
        Shape: `(B, T, 2D)` where `D = max_action_dim`
    next_mu : chex.Array
        Actual policy mean at next timestep. Shape: `(B, T, D)`
    next_log_std : chex.Array
        Actual policy log-std at next timestep. Shape: `(B, T, D)`
    discounts : chex.Array
        Episode continuation signals. Shape: `(B, T, 1)`
    action_dim_mask : chex.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    loss : chex.Array
        Per-timestep auxiliary policy loss. Shape: `(B, T-1)`
    """
    max_action_dim = jnp.shape(aux_pi_pred)[-1] // 2  # Static: 2D layout is (μ', log σ')

    # Split predicted (μ', log σ') from aux_pi output
    pred_mu = aux_pi_pred[:, :-1, :max_action_dim]  # (B, T-1, D), # type: ignore
    pred_log_std = aux_pi_pred[:, :-1, max_action_dim:]  # (B, T-1, D), # type: ignore

    # Actual next-step policy (stop gradient — this is the target)
    target_mu = jax.lax.stop_gradient(next_mu[:, 1:])  # (B, T-1, D), # type: ignore
    target_log_std = jax.lax.stop_gradient(
        next_log_std[:, 1:]  # type: ignore
    )  # (B, T-1, D)

    # Gaussian KL between predicted and actual next-step policy
    loss = compute_gaussian_kl(
        target_mu,
        target_log_std,
        pred_mu,
        pred_log_std,
        action_dim_mask=action_dim_mask,
    )

    # Mask out terminal states
    return loss * jnp.squeeze(discounts[:, :-1], axis=-1)  # (B, T-1), # type: ignore
