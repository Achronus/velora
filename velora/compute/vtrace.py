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

from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp


class VTraceOutput(NamedTuple):
    """
    Output of `vtrace_td_error_and_advantage`.

    Attributes
    ----------
    errors : chex.Array
        V-trace TD errors `(v_s - V(x_s))`. Shape: `(T,)`
    pg_advantage : chex.Array
        Policy-gradient advantages. Shape: `(T,)`
    q_estimate : chex.Array
        Off-policy Q-value estimates. Shape: `(T,)`
    """

    errors: chex.Array
    pg_advantage: chex.Array
    q_estimate: chex.Array


def _vtrace_errors(
    v_tm1: chex.Array,
    v_t: chex.Array,
    r_t: chex.Array,
    discount_t: chex.Array,
    rho_tm1: chex.Array,
    lambda_: float,
    clip_rho_threshold: float,
) -> chex.Array:
    """Compute raw V-trace correction terms via a backward scan."""
    clipped_rho = jnp.minimum(clip_rho_threshold, rho_tm1)
    c_t = jnp.minimum(clip_rho_threshold, rho_tm1) * lambda_

    # Per-step TD error weighted by clipped importance ratio
    delta = clipped_rho * (r_t + discount_t * v_t - v_tm1)  # type: ignore

    def _scan_fn(acc, xs):
        delta_t, discount_t, c_t = xs
        acc = delta_t + discount_t * c_t * acc
        return acc, acc

    _, errors = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(delta[-1]),
        (delta, discount_t, c_t),
        reverse=True,
    )
    return errors


def vtrace_td_error_and_advantage(
    v_tm1: chex.Array,
    v_t: chex.Array,
    r_t: chex.Array,
    discount_t: chex.Array,
    rho_tm1: chex.Array,
    lambda_: float = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> VTraceOutput:
    """
    V-Trace TD errors and policy-gradient advantages (IMPALA).

    Implements the off-policy correction algorithm from Espeholt et al.
    (2018). All inputs are **single-trajectory** (no batch dim); use
    `jax.vmap` to handle batches, matches the `rlax` convention.

    Parameters
    ----------
    v_tm1 : chex.Array
        State values at `t`. Shape: `(T,)`
    v_t : chex.Array
        State values at `t+1`. Shape: `(T,)`
    r_t : chex.Array
        Rewards at `t`. Shape: `(T,)`
    discount_t : chex.Array
        Discounts (`γ * (1 - done)`) at `t`. Shape: `(T,)`
    rho_tm1 : chex.Array
        Importance weights `π(a|s) / μ(a|s)` at `t`. Shape: `(T,)`
    lambda_ : float
        Trace-decay parameter. Default is `1.0`.
    clip_rho_threshold : float
        Clipping threshold for importance weights in value targets.
        Default is `1.0`.
    clip_pg_rho_threshold : float
        Clipping threshold for importance weights in PG advantages.
        Default is `1.0`.
    stop_target_gradients : bool
        Whether to stop gradients through the value targets.
        Default is `True`.

    Returns
    -------
    output : VTraceOutput
        Named tuple of `(errors, pg_advantage, q_estimate)`.
    """
    errors = _vtrace_errors(
        v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_, clip_rho_threshold
    )

    # V-trace corrected values: v_s = V(x_s) + errors
    vs = errors + v_tm1

    if stop_target_gradients:
        errors = jax.lax.stop_gradient(errors)
        vs = jax.lax.stop_gradient(vs)

    # Shift targets forward: vs_{t+1} for each step
    # Last step bootstraps from v_t[-1] (the final next-state value)
    vs_t_plus_1 = jnp.concatenate([vs[1:], v_t[-1:]], axis=0)  # type: ignore

    # Policy-gradient advantage: ρ̄ * (r + γ·v_{s+1} - V(x_s))
    clipped_pg_rho = jnp.minimum(clip_pg_rho_threshold, rho_tm1)
    pg_advantage = clipped_pg_rho * (r_t + discount_t * vs_t_plus_1 - v_tm1)

    # Q-value estimate
    q_estimate = r_t + discount_t * vs_t_plus_1

    return VTraceOutput(
        errors=errors,
        pg_advantage=pg_advantage,
        q_estimate=q_estimate,
    )


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
        return vtrace_td_error_and_advantage(
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
