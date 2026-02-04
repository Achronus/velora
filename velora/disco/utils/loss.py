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

from typing import TYPE_CHECKING, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
import rlax

if TYPE_CHECKING:
    from velora.disco.outputs import AgentLosses, DiscoAgentOutput
    from velora.disco.settings import LossCostSettings


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


def compute_policy_gradient_loss(
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
    log_pi_a = rlax.batched_index(log_pi, actions)
    return -log_pi_a * jax.lax.stop_gradient(advantages)  # type: ignore


def compute_meta_reg_loss(
    targets: "DiscoAgentOutput",
    target_pi: chex.Array,
    reg_scale: float,
    kl_reg_coef: float,
) -> chex.Array:
    """
    Compute regularization loss for meta-network targets.

    Includes:
        - L2 regularization on target means (prevent divergence)
        - KL divergence between targets and current policy (stability)

    Parameters
    ----------
    targets : DiscoAgentOutput
        Generated targets `(π̂, ŷ, ẑ)`
    target_pi : chex.Array
        Stop-gradiented target policy logits for KL computation
    reg_scale : float
        L2 regularization scale
    kl_reg_coef : float
        KL regularization coefficient

    Returns
    -------
    reg_loss : chex.Array
        Total regularization loss
    """
    from velora.disco.utils.compute import compute_l2_mean_penalty

    pi_reg = compute_l2_mean_penalty(targets.pi)
    y_reg = compute_l2_mean_penalty(targets.y)
    z_reg = compute_l2_mean_penalty(targets.z)

    target_kl = compute_kl_loss(
        jax.lax.stop_gradient(target_pi),
        targets.pi,
    ).mean()

    l2_loss = reg_scale * (pi_reg + y_reg + z_reg)
    kl_loss = kl_reg_coef * target_kl
    return l2_loss + kl_loss


def compute_policy_loss(
    targets: "DiscoAgentOutput",
    preds_pi: chex.Array,
    preds_y: chex.Array,
    preds_z: chex.Array,
    preds_aux_pi: chex.Array,
    actions: chex.Array,
    discounts: chex.Array,
    loss_costs: "LossCostSettings",
) -> Tuple[chex.Array, "AgentLosses"]:
    """
    Compute policy agent losses for training against disco targets.

    The loss consists of:
        - KL divergence between target policy (`π̂`) and agent policy (`π`)
        - KL divergence between target y (`ŷ`) and agent `y`
        - KL divergence between target z (`ẑ`) and agent `z` (for action taken)
        - Auxiliary policy prediction loss

    Parameters
    ----------
    targets : DiscoAgentOutput
        Targets generated by the meta-network `(π̂, ŷ, ẑ)`
    preds_pi : chex.Array
        Agent policy predictions. Shape: `(B, T, A)`
    preds_y : chex.Array
        Agent y predictions. Shape: `(B, T, Y)`
    preds_z : chex.Array
        Agent z predictions. Shape: `(B, T, A, Z)`
    preds_aux_pi : chex.Array
        Agent auxiliary policy predictions. Shape: `(B, T, A, A)`
    actions : chex.Array
        Actions taken. Shape: `(B, T, 1)`
    discounts : chex.Array
        Episode continuation signals. Shape: `(B, T, 1)`
    loss_costs : LossCostSettings
        Loss weighting coefficients

    Returns
    -------
    total_loss : chex.Array
        Total weighted loss
    losses : AgentLosses
        Individual loss components
    """
    from velora.disco.outputs import AgentLosses

    pi_loss = compute_kl_loss(targets.pi, preds_pi).mean()
    y_loss = compute_kl_loss(targets.y, preds_y).mean()
    z_loss = compute_z_loss(preds_z, targets.z, actions).mean()
    aux_pi_loss = compute_aux_policy_loss(
        preds_aux_pi,
        preds_pi,
        actions,
        discounts,
    ).mean()

    losses = AgentLosses(
        pi=pi_loss,
        y=y_loss,
        z=z_loss,
        aux_pi=aux_pi_loss,
    ).compute_total(loss_costs)

    return losses.total, losses
