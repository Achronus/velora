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

from dataclasses import fields
from typing import Any, Dict, Self

import chex
import jax
from flax import struct

from velora.disco.outputs import PolicyAgentOutput
from velora.utils.transforms import to_time_first


@struct.dataclass
class BufferSamples:
    """
    Dataclass for a batch of trajectories sampled from the buffer.

    Parameters
    ----------
    actions : jax.Array
        Actions taken with shape `(B, T, 1)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory.

    rewards : jax.Array
        Rewards received with shape `(B, T, 1)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory.

    discounts : jax.Array
        Raw environment discounts (dones) with shape `(B, T, 1)`. Binary values
        where `1.0` = episode continues, `0.0` = episode ended. Multiply by
        gamma at training time:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory.

    values : jax.Array
        State-value estimates `V(s)` from the value network with shape `(B, T, 1)`.

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
    values: chex.Array

    preds: PolicyAgentOutput
    target_preds: PolicyAgentOutput

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary.

        Useful for unpacking a trajectory of samples into `buffer.add()`.

        Returns
        -------
        dict : Dict[str, Any]
            Object in dictionary format
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_time_first(self) -> Self:
        """
        Transpose batch from batch-first to time-first format.

        Converts shape from `(B, T, ...)` to `(T, B, ...)` for all array fields.
        Useful for V-trace and other temporal computations that expect time
        as the leading dimension.

        Returns
        -------
        samples : BufferSamples
            New instance with time-first arrays
        """

        return self.__replace__(
            actions=to_time_first(self.actions),
            rewards=to_time_first(self.rewards),
            discounts=to_time_first(self.discounts),
            values=to_time_first(self.values),
            preds=jax.tree.map(to_time_first, self.preds),
            target_preds=jax.tree.map(to_time_first, self.target_preds),
        )
