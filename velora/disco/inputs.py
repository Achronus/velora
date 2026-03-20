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

from typing import NamedTuple

import chex
import optax

from velora.disco.ema import EMAState
from velora.disco.rollouts import Rollout


class MetaLossFnInputs(NamedTuple):
    """
    Non-differentiated inputs for the JIT-compiled meta-loss function.

    Parameters
    ----------
    policy_params : optax.Params
        Policy agent parameters
    value_params : optax.Params
        Value agent parameters
    disco_h : chex.Array
        Disco network hidden state
    meta_h : chex.Array
        Meta LNN hidden state
    p_opt_state : optax.OptState
        Policy optimizer state
    v_opt_state : optax.OptState
        Value optimizer state
    adv_ema : EMAState
        Advantage EMA state
    td_ema : EMAState
        TD-error EMA state
    train_rollouts : Rollout
        Training rollouts for the inner loop
    valid_rollout : Rollout
        Validation rollout for the meta-loss
    """

    policy_params: optax.Params
    value_params: optax.Params
    disco_h: chex.Array
    meta_h: chex.Array
    p_opt_state: optax.OptState
    v_opt_state: optax.OptState
    adv_ema: EMAState
    td_ema: EMAState
    train_rollouts: Rollout
    valid_rollout: Rollout
