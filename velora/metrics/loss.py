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
import distrax
import jax
import jax.numpy as jnp
import rlax


def compute_kl_loss(
    target: chex.Array,
    preds: chex.Array,
) -> chex.Array:
    """
    Compute KL divergence loss between target and predicted logits.

    Uses softmax to convert logits to probabilities internally.

    Parameters
    ----------
    target : chex.Array
        Target logits from the meta-network
    preds : chex.Array
        Predicted logits from the policy agent

    Returns
    -------
    loss : chex.Array
        Per-element KL divergence loss
    """
    return rlax.categorical_kl_divergence(target, preds)


def compute_z_loss(
    z: chex.Array,
    z_targets: chex.Array,
    actions: chex.Array,
) -> chex.Array:
    """
    Compute KL divergence loss for goal distribution predictions.

    Extracts the predicted goal distribution for the actions taken and
    computes KL divergence against the target goal distribution.

    Parameters
    ----------
    z : chex.Array
        Predicted goal distributions for all actions `(B, T, A, Z)`
    z_targets : chex.Array
        Target goal distributions `(B, T, Z)`
    actions : chex.Array
        Actions taken at each timestep `(B, T, 1)`

    Returns
    -------
    loss : chex.Array
        Per-timestep goal distribution loss `(B, T)`
    """
    action_preds = jnp.take_along_axis(
        z,
        actions[..., None],  # (B, T, 1, 1), # type: ignore
        axis=2,
    ).squeeze(2)  # (B, T, Z)

    return compute_kl_loss(z_targets, action_preds)


def compute_aux_policy_loss(
    aux_pi: chex.Array,
    next_pi: chex.Array,
    actions: chex.Array,
    discounts: chex.Array,
) -> chex.Array:
    """
    Compute auxiliary 1-step policy prediction loss.

    The auxiliary policy `p(s, a)` predicts what the policy will be at the
    next timestep, masked by episode boundaries.

    Parameters
    ----------
    aux_pi : chex.Array
        Auxiliary policy predictions for action taken `(B, T, A, A)`
    next_pi : chex.Array
        Actual policy at next timestep `(B, T, A)`
    actions : chex.Array
        Actions at next timestep `(B, T, 1)`
    discounts : chex.Array
        Episode continuation signals `(B, T, 1)`

    Returns
    -------
    loss : chex.Array
        Per-timestep auxiliary policy loss `(B, T)`
    """
    # Predict next timesteps policy
    aux_pi_a = jnp.take_along_axis(
        aux_pi[:, :-1],  # (B, T-1, A, A), # type: ignore
        actions[:, :-1, ..., None],  # (B, T-1, 1, 1), # type: ignore
        axis=2,
    )

    loss = compute_kl_loss(
        jax.lax.stop_gradient(next_pi[:, 1:]),  # (B, T-1, A), # type: ignore
        jnp.squeeze(aux_pi_a, axis=2),  # (B, T-1, A)
    )

    # Mask out terminal states
    return loss * jnp.squeeze(discounts[:, :-1], axis=-1)  # type: ignore


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
    return -coef * jnp.mean(distrax.Softmax(logits).entropy())
