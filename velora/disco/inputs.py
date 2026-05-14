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

import jax
import optax

from velora.disco.ema import EMAState
from velora.disco.rollouts import Rollout


class GradChunkInputs(NamedTuple):
    """
    Sliced inputs for a single chunk of the vmapped gradient function.

    All fields have a leading chunk dimension `(C, ...)` where
    `C = end - start` trainers.

    Parameters
    ----------
    p_params : optax.Params
        Policy network parameters
    v_params : optax.Params
        Value network parameters
    disco_h : jax.Array
        Disco network hidden states `(C, B, H)`
    meta_h : jax.Array
        Meta-LNN hidden states `(C, B, H)`
    p_opt : optax.OptState
        Policy optimizer states
    v_opt : optax.OptState
        Value optimizer states
    adv_ema : EMAState
        Advantage EMA states
    td_ema : EMAState
        TD-error EMA states
    train_rollout : Rollout
        Training rollouts `(C, N, B, T, ...)`
    valid_rollout : Rollout
        Validation rollouts `(C, 1, B, T, ...)`
    masks : jax.Array
        Boolean masks `(C, max_action_size)` for valid actions/dims
    """

    p_params: optax.Params
    v_params: optax.Params
    disco_h: jax.Array
    meta_h: jax.Array
    p_opt: optax.OptState
    v_opt: optax.OptState
    adv_ema: EMAState
    td_ema: EMAState
    train_rollout: Rollout
    valid_rollout: Rollout
    masks: jax.Array
