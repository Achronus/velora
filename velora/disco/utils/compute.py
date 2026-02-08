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

from typing import Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
import rlax

from velora.disco.ema import EMAState, MovingAverage
from velora.disco.outputs import ValueOutputs
from velora.disco.rollouts import Rollout


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
    log_pi = distrax.Softmax(pi_logits).log_prob(actions)
    log_mu = distrax.Softmax(mu_logits).log_prob(actions)
    rho = jax.lax.stop_gradient(jnp.exp(log_pi - log_mu))  # type: ignore
    return rho


def compute_vtrace(
    values: chex.Array,
    rewards: chex.Array,
    discounts: chex.Array,
    td_lambda: float,
    rho: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """
    Compute V-trace targets and advantages.

    Parameters
    ----------
    values : chex.Array
        State value estimates. Shape: `(T+1, B)`
    rewards : chex.Array
        Rewards. Shape: `(T, B)`
    discounts : chex.Array
        Discount factors multiplied by gamma (0 at episode end). Shape: `(T, B)`
    td_lambda : float
        The `λ` used for computing advantage estimates.

        When -
        - `λ=0.0` → Use only 1-step TD (immediate reward + bootstrap)
        - `λ=1.0` → Use full Monte Carlo return (entire episode)
        - `λ=0.95` → Blend of n-step returns (weighted toward longer horizons)
    rho : chex.Array
        Importance weights. Shape: `(T, B)`

    Returns
    -------
    value_targets : chex.Array
        V-trace value targets. Shape: `(T, B)`
    advantages : chex.Array
        V-trace advantages. Shape: `(T, B)`
    """

    # V-trace computation (vmapped over batch dimension)
    def vtrace_single(v, v_next, r, d, rho_t):
        return rlax.vtrace_td_error_and_advantage(
            v_tm1=v,
            v_t=v_next,
            r_t=r,
            discount_t=d,
            rho_tm1=rho_t,
            lambda_=td_lambda,
        )

    vtrace_fn = jax.vmap(vtrace_single, in_axes=1, out_axes=1)
    vtrace_out = vtrace_fn(values[:-1], values[1:], rewards, discounts, rho)  # type: ignore

    value_targets = vtrace_out.errors + values[:-1]  # type: ignore
    advantages = vtrace_out.pg_advantage

    return value_targets, advantages


def compute_l2_mean_penalty(x: chex.Array) -> chex.Array:
    """
    Compute L2 penalty on the mean of a tensor.

    Penalizes the global mean drifting from zero, encouraging
    centered outputs. Useful for regularizing meta-network targets
    to prevent divergence.

    Parameters
    ----------
    x : chex.Array
        Input tensor of any shape

    Returns
    -------
    penalty : chex.Array
        Scalar L2 penalty on the mean
    """
    return jnp.square(jnp.mean(x))


def compute_value_outputs(
    rollout: Rollout,
    ema_utils: MovingAverage,
    adv_state: EMAState,
    td_state: EMAState,
    gamma: float,
    td_lambda: float,
) -> Tuple[ValueOutputs, EMAState, EMAState]:
    """
    Compute value function outputs from a trajectory of experience.

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

    Returns
    -------
    value_outs : ValueOutputs
        Value function outputs
    adv_ema : EMAState
        Updated advantage EMA state
    td_ema : EMAState
        Updated TD EMA state
    """
    # Transpose to (T, B) for V-trace
    rollout = rollout.to_time_first()
    rollout = rollout.squeeze()

    discounts = rollout.discounts * gamma

    # [:-1] = Drop last timestep
    rho = compute_importance_weights(
        rollout.preds.pi[:-1],  # type: ignore
        rollout.target_preds.pi[:-1],  # type: ignore
        rollout.actions[:-1],  # type: ignore
    )

    value_targets, advantages = compute_vtrace(
        rollout.values,
        rollout.rewards[:-1],  # type: ignore
        discounts[:-1],  # type: ignore
        td_lambda,
        rho,
    )

    td = value_targets - rollout.values[:-1]  # type: ignore

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
