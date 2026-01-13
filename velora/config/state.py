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
from flax import struct


@struct.dataclass
class AgentHiddenStates:
    """
    Dataclass for agent hidden states.

    Parameters
    ----------
    ocm : jax.Array
        Observation-Conditional Model (OCM) hidden states `(B, H)`

        - `batch_size (B)` the number of samples per timestep
        - `n_units (H)` the total number of OCM hidden neurons

    acm : jax.Array
        Action-Conditional Model (ACM) hidden states `(B * A, H)`

        - `batch_size (B)` the number of samples per timestep
        - `n_actions (A)` the number of discrete actions
        - `n_units (H)` the total number of ACM hidden neurons
    """

    ocm: chex.Array
    acm: chex.Array
