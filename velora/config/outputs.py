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

from velora.config.state import AgentHiddenStates


@struct.dataclass
class AgentOutput:
    """
    Storage container for the `VeloraAgent` output.

    Parameters:
        pi (jax.Array): policy logits `(B, A, T)`

            - `batch_size (B)` the number of samples per timestep.
            - `n_actions (A)` the number of discrete actions in the action space.
            - `seq_length (T)` the number of sequences (e.g., trajectories).

        y (jax.Array): observation-conditioned prediction vector `(B, Y, T)`

            - `batch_size (B)` the number of samples per timestep.
            - `y_dim (Y)` the size of the observation-conditioned prediction
                vector.
            - `seq_length (T)` the number of sequences (e.g., trajectories).

        z (jax.Array): action-conditioned prediction vector `(B, A, Z, T)`

            - `batch_size (B)` the number of samples per timestep.
            - `n_actions (A)` the number of discrete actions in the action space.
            - `z_dim (Z)` the size of the action-conditioned prediction vector.
            - `seq_length (T)` the number of sequences (e.g., trajectories).

        aux_pi (jax.Array): auxiliary policy logits `(B, A, A, T)`

                - `batch_size (B)` the number of samples per timestep.
                - `n_actions (A)` the number of discrete actions in the action
                    space.
                - `seq_length (T)` the number of sequences (e.g., trajectories).

        q (jax.Array): action-value predictions `(B, A, Q, T)`

            - `batch_size (B)` the number of samples per timestep.
            - `n_actions (A)` the number of discrete actions in the action space.
            - `q_dim (Q)` the size of the action-value prediction head.
            - `seq_length (T)` the number of sequences (e.g., trajectories).

        embedding (jax.Array): agent embedding vector `(B, E, T)`

            - `batch_size (B)` the number of samples per timestep.
            - `embedding_dim (E)` the size of the agent embedding vector.
            - `seq_length (T)` the number of sequences (e.g., trajectories).

        h_states (AgentHiddenStates): the agent hidden states as an object
    """

    pi: chex.Array
    y: chex.Array
    z: chex.Array
    aux_pi: chex.Array
    q: chex.Array
    embedding: chex.Array
    h_states: AgentHiddenStates
