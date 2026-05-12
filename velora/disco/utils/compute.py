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

from velora.compute.rl_ops import compute_gaussian_importance_weights
from velora.compute.vtrace import compute_vtrace
from velora.disco.ema import EMAState, MovingAverage
from velora.disco.outputs import ValueOutputs
from velora.disco.rollouts import Rollout


def compute_value_outputs(
    rollout: Rollout,
    ema_utils: MovingAverage,
    adv_state: EMAState,
    td_state: EMAState,
    gamma: float,
    td_lambda: float,
    action_dim_mask: chex.Array | None = None,
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
    action_dim_mask : chex.Array (optional)
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
        rollout.preds.mu[:-1],  # type: ignore
        rollout.preds.log_std[:-1],  # type: ignore
        rollout.target_preds.mu[:-1],  # type: ignore
        rollout.target_preds.log_std[:-1],  # type: ignore
        rollout.actions[:-1],  # type: ignore
        action_dim_mask=action_dim_mask,
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
