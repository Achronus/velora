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
class AgentOutput:
    """
    Storage container for the `VeloraAgent` output.

    Parameters
    ----------
    pi : jax.Array
        Policy logits with shape `(B, T, A)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of sequences (e.g., trajectories).
        - n_actions (`A`) - the number of discrete actions in the action space.

    y : jax.Array
        Observation-conditioned prediction vector with shape `(B, T, Y)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of sequences (e.g., trajectories).
        - y_dim (`Y`) - the size of the observation-conditioned prediction vector.

    z : jax.Array
        Action-conditioned prediction vector with shape `(B, T, A, Z)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of sequences (e.g., trajectories).
        - n_actions (`A`) - the number of discrete actions in the action space.
        - z_dim (`Z`) - the size of the action-conditioned prediction vector.

    aux_pi : jax.Array
        Auxiliary policy logits with shape `(B, T, A, A)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of sequences (e.g., trajectories).
        - n_actions (`A`) - the number of discrete actions in the action space.

    q : jax.Array
        Action-value predictions with shape `(B, T, A, Q)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of sequences (e.g., trajectories).
        - n_actions (`A`) - the number of discrete actions in the action space.
        - q_dim (`Q`) - the size of the action-value prediction head.
    """

    pi: chex.Array
    y: chex.Array
    z: chex.Array
    aux_pi: chex.Array
    q: chex.Array


@struct.dataclass
class BufferSamples:
    """
    A batch of trajectories sampled from the buffer.

    Parameters
    ----------
    actions : jax.Array
        Actions taken with shape `(B, T)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory.

    rewards : jax.Array
        Rewards received with shape `(B, T)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory.

    discounts : jax.Array
        Raw environment discounts (dones) with shape `(B, T)`. Binary values
        where `1.0` = episode continues, `0.0` = episode ended. Multiply by
        gamma at training time:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory.

    preds : AgentOutput
        Agent network output predictions at each timestep

    target_preds : AgentOutput
        Agent target network output predictions at each timestep
    """

    actions: chex.Array
    rewards: chex.Array
    discounts: chex.Array
    preds: AgentOutput
    target_preds: AgentOutput
