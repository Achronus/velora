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
import jax
import jax.numpy as jnp

from velora.compute.loss import compute_gaussian_aux_policy_loss
from velora.compute.rl_ops import compute_gaussian_kl
from velora.compute.utils import categorical_kl_divergence

if TYPE_CHECKING:
    from velora.disco.config.settings import LossCostSettings
    from velora.disco.outputs import (
        AgentLosses,
        ContinuousDiscoAgentOutput,
        DiscoAgentOutput,
    )


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

    return categorical_kl_divergence(z_targets, action_preds)


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

    loss = categorical_kl_divergence(
        jax.lax.stop_gradient(next_pi[:, 1:]),  # (B, T-1, A), # type: ignore
        jnp.squeeze(aux_pi_a, axis=2),  # (B, T-1, A)
    )

    # Mask out terminal states
    return loss * jnp.squeeze(discounts[:, :-1], axis=-1)  # type: ignore


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
    from velora.compute.utils import compute_l2_mean_penalty

    pi_reg = compute_l2_mean_penalty(targets.pi)
    y_reg = compute_l2_mean_penalty(targets.y)
    z_reg = compute_l2_mean_penalty(targets.z)

    target_kl = categorical_kl_divergence(
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

    pi_loss = categorical_kl_divergence(targets.pi, preds_pi).mean()
    y_loss = categorical_kl_divergence(targets.y, preds_y).mean()
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


def compute_continuous_policy_loss(
    targets: "ContinuousDiscoAgentOutput",
    preds_mu: chex.Array,
    preds_log_std: chex.Array,
    preds_y: chex.Array,
    preds_z: chex.Array,
    preds_aux_pi: chex.Array,
    discounts: chex.Array,
    loss_costs: "LossCostSettings",
    action_dim_mask: chex.Array | None = None,
) -> Tuple[chex.Array, chex.Array]:
    """
    Compute policy agent losses for training against disco targets.

    The loss consists of:
        - Gaussian KL divergence between target policy `(μ̂, log σ̂)` and
          agent policy `(μ, log σ)`
        - Categorical KL divergence between target `ŷ` and agent `y`
        - Categorical KL divergence between target `ẑ` and agent `z`
        - Auxiliary 1-step Gaussian policy prediction loss

    Parameters
    ----------
    targets : ContinuousDiscoAgentOutput
        Targets generated by the meta-network `(μ̂, log σ̂, ŷ, ẑ)`
    preds_mu : chex.Array
        Agent policy mean. Shape: `(B, T, D)`
    preds_log_std : chex.Array
        Agent policy log-std. Shape: `(B, T, D)`
    preds_y : chex.Array
        Agent y predictions. Shape: `(B, T, Y)`
    preds_z : chex.Array
        Agent z predictions. Shape: `(B, T, Z)`
    preds_aux_pi : chex.Array
        Agent auxiliary policy predictions. Shape: `(B, T, 2D)`
    discounts : chex.Array
        Episode continuation signals. Shape: `(B, T, 1)`
    loss_costs : LossCostSettings
        Loss weighting coefficients
    action_dim_mask : chex.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    total_loss : chex.Array
        Total weighted loss
    pi_loss : chex.Array
        Policy KL divergence component (for logging)
    """
    # Policy loss: Gaussian KL
    pi_loss = compute_gaussian_kl(
        targets.mu,
        targets.log_std,
        preds_mu,
        preds_log_std,
        action_dim_mask=action_dim_mask,
    ).mean()

    # y loss: categorical KL (softmax-normalized learned vectors)
    y_loss = categorical_kl_divergence(targets.y, preds_y).mean()

    # z loss: categorical KL (already conditioned on action — no indexing)
    z_loss = categorical_kl_divergence(targets.z, preds_z).mean()

    # Auxiliary policy loss: predict next-step Gaussian params
    aux_pi_loss = compute_gaussian_aux_policy_loss(
        preds_aux_pi,
        preds_mu,
        preds_log_std,
        discounts,
        action_dim_mask=action_dim_mask,
    ).mean()

    total_loss = (
        loss_costs.pi * pi_loss
        + loss_costs.y * y_loss
        + loss_costs.z * z_loss
        + loss_costs.aux_pi * aux_pi_loss
    )

    return total_loss, pi_loss


def compute_continuous_meta_reg_loss(
    targets: "ContinuousDiscoAgentOutput",
    target_mu: chex.Array,
    target_log_std: chex.Array,
    reg_scale: float,
    kl_reg_coef: float,
    action_dim_mask: chex.Array | None = None,
) -> chex.Array:
    """
    Compute regularization loss for continuous meta-network targets.

    Includes:
        - L2 regularization on target means (prevent divergence)
        - Gaussian KL between targets and current policy (stability)

    Parameters
    ----------
    targets : ContinuousDiscoAgentOutput
        Generated targets `(μ̂, log σ̂, ŷ, ẑ)`
    target_mu : chex.Array
        Stop-gradiented target policy mean for KL computation
    target_log_std : chex.Array
        Stop-gradiented target policy log-std for KL computation
    reg_scale : float
        L2 regularization scale
    kl_reg_coef : float
        KL regularization coefficient
    action_dim_mask : chex.Array (optional)
        Boolean mask for valid action dimensions. Default is `None`

    Returns
    -------
    reg_loss : chex.Array
        Total regularization loss
    """
    from velora.compute import compute_l2_mean_penalty

    mu_reg = compute_l2_mean_penalty(targets.mu)
    log_std_reg = compute_l2_mean_penalty(targets.log_std)
    y_reg = compute_l2_mean_penalty(targets.y)
    z_reg = compute_l2_mean_penalty(targets.z)

    target_kl = compute_gaussian_kl(
        jax.lax.stop_gradient(target_mu),
        jax.lax.stop_gradient(target_log_std),
        targets.mu,
        targets.log_std,
        action_dim_mask=action_dim_mask,
    ).mean()

    l2_loss = reg_scale * (mu_reg + log_std_reg + y_reg + z_reg)
    kl_loss = kl_reg_coef * target_kl
    return l2_loss + kl_loss
