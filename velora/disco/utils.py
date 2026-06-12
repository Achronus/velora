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
from flax import nnx

from velora.compute.loss import compute_gaussian_aux_policy_loss
from velora.compute.rl_ops import (
    compute_gaussian_importance_weights,
    compute_gaussian_kl,
)
from velora.compute.utils import categorical_kl_divergence
from velora.compute.vtrace import compute_vtrace
from velora.disco.ema import EMAState, MovingAverage
from velora.disco.rollouts import Rollout

if TYPE_CHECKING:
    from velora.disco.config.settings import LossCostSettings
    from velora.disco.outputs import DiscoAgentOutput, ValueOutputs


def sample_budget(key: chex.PRNGKey) -> jax.Array:
    """
    Sample a step budget from a lifetime distribution.

    Budgets are drawn from `{5M, 10M, 20M, 50M}` environment steps,
    weighted inversely proportional to their size so shorter lifetimes are
    sampled more frequently.

    Parameters
    ----------
    key : chex.PRNGKey
        Random number generator key

    Returns
    -------
    budget : jax.Array
        Sampled step budget as a scalar `int32`
    """
    lifetime_budgets = jnp.array(
        [5_000_000, 10_000_000, 20_000_000, 50_000_000],
        dtype=jnp.int32,
    )
    _inv = 1.0 / lifetime_budgets.astype(jnp.float32)
    lifetime_weights = _inv / _inv.sum()

    return jax.random.choice(key, lifetime_budgets, p=lifetime_weights)


def compute_value_outputs(
    rollout: Rollout,
    ema_utils: MovingAverage,
    adv_state: EMAState,
    td_state: EMAState,
    gamma: float,
    td_lambda: float,
    action_dim_mask: jax.Array | None = None,
) -> Tuple[ValueOutputs, EMAState, EMAState]:
    """
    Compute value function outputs from a trajectory of experience
    using Gaussian importance weights for off-policy correction.

    Parameters
    ----------
    rollout : Rollout
        A trajectory of experience
    ema_utils : MovingAverage
        EMA utility methods for computation
    adv_state : EMAState
        EMA state for advantage normalization
    td_state : EMAState
        EMA state for TD normalization
    gamma : float
        Discount factor
    td_lambda : float
        TD lambda parameter
    action_dim_mask : jax.Array (optional)
        Boolean mask `(D,)` for valid action dimensions.
        Default is `None`

    Returns
    -------
    value_outs : ValueOutputs
        Value function outputs
    adv_ema : EMAState
        Updated advantage EMA state
    td_ema : EMAState
        Updated TD EMA state
    """

    # Transpose to (T, B, ...) for V-trace
    rollout = rollout.to_time_first()
    rollout = rollout.squeeze()

    discounts = rollout.discounts * gamma

    # Importance weights from Gaussian policies
    # [:-1] = Drop last timestep
    rho = compute_gaussian_importance_weights(
        rollout.preds.mu[:-1],
        rollout.preds.log_std[:-1],
        rollout.target_preds.mu[:-1],
        rollout.target_preds.log_std[:-1],
        rollout.actions[:-1],
        action_dim_mask=action_dim_mask,
    )

    value_targets, advantages = compute_vtrace(
        rollout.values,
        rollout.rewards[:-1],
        discounts[:-1],
        td_lambda,
        rho,
    )

    td = value_targets - rollout.values[:-1]

    # Compute EMAs
    norm_adv, adv_state = ema_utils.update_and_normalize(advantages, adv_state)
    norm_td, td_state = ema_utils.update_and_normalize(
        td,
        td_state,
        subtract_mean=False,
    )

    value_outs = ValueOutputs(
        value=rollout.values,
        value_targets=value_targets,
        advantages=advantages,
        normalized_advantages=norm_adv,
        td=td,
        normalized_td=norm_td,
        rho=rho,
    )

    return value_outs, adv_state, td_state
    return value_outs, adv_state, td_state


def compute_policy_loss(
    targets: "DiscoAgentOutput",
    preds_mu: jax.Array,
    preds_log_std: jax.Array,
    preds_y: jax.Array,
    preds_z: jax.Array,
    preds_aux_pi: jax.Array,
    discounts: jax.Array,
    loss_costs: "LossCostSettings",
    action_dim_mask: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
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
    targets : DiscoAgentOutput
        Targets generated by the meta-network `(μ̂, log σ̂, ŷ, ẑ)`
    preds_mu : jax.Array
        Agent policy mean. Shape: `(B, T, D)`
    preds_log_std : jax.Array
        Agent policy log-std. Shape: `(B, T, D)`
    preds_y : jax.Array
        Agent y predictions. Shape: `(B, T, Y)`
    preds_z : jax.Array
        Agent z predictions. Shape: `(B, T, Z)`
    preds_aux_pi : jax.Array
        Agent auxiliary policy predictions. Shape: `(B, T, 2D)`
    discounts : jax.Array
        Episode continuation signals. Shape: `(B, T, 1)`
    loss_costs : LossCostSettings
        Loss weighting coefficients
    action_dim_mask : jax.Array (optional)
        Boolean mask `(D,)` for valid action dimensions. Default is `None`

    Returns
    -------
    total_loss : jax.Array
        Total weighted loss
    pi_loss : jax.Array
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


def compute_meta_reg_loss(
    targets: "DiscoAgentOutput",
    target_mu: jax.Array,
    target_log_std: jax.Array,
    reg_scale: float,
    kl_reg_coef: float,
    action_dim_mask: jax.Array | None = None,
) -> jax.Array:
    """
    Compute regularization loss for meta-network targets.

    Includes:
        - L2 regularization on target means (prevent divergence)
        - Gaussian KL between targets and current policy (stability)

    Parameters
    ----------
    targets : DiscoAgentOutput
        Generated targets `(μ̂, log σ̂, ŷ, ẑ)`
    target_mu : jax.Array
        Stop-gradiented target policy mean for KL computation
    target_log_std : jax.Array
        Stop-gradiented target policy log-std for KL computation
    reg_scale : float
        L2 regularization scale
    kl_reg_coef : float
        KL regularization coefficient
    action_dim_mask : jax.Array (optional)
        Boolean mask for valid action dimensions. Default is `None`

    Returns
    -------
    reg_loss : jax.Array
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


def soft_param_update(
    tau: float, old_params: nnx.State, new_params: nnx.State
) -> nnx.State:
    """
    Performs a soft parameter update on a set of network parameters.

    Formula: `θ_target ← τ * θ_online + (1 - τ) * θ_target`

    Parameters
    ----------
    tau : float
        Soft update coefficient (`τ`)
    old_params : nnx.State
        Current network parameters (`θ_online`)
    new_params : nnx.State
        New parameters to use for updating (`θ_target`)

    Returns
    -------
    params : nnx.State
        An updated set of network parameters
    """
    return jax.tree.map(
        lambda old, new: tau * old + (1.0 - tau) * new,
        old_params,
        new_params,
    )
