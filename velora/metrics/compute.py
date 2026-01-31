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
