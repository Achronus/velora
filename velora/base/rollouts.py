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

from typing import List, Self

import chex
import jax
import jax.numpy as jnp
from flax import struct

from velora.disco.outputs import PolicyAgentOutput
from velora.utils.structs import get_fields_by_index
from velora.utils.transforms import to_time_first


@struct.dataclass
class Rollout:
    """
    A single agent trajectory.

    Parameters
    ----------
    actions : jax.Array
        Actions taken in the environment `(B, T, 1)`

        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) the number of timesteps in the trajectory (can be more than one episode)

    rewards : jax.Array
        Rewards generated from the environment `(B, T, 1)`

        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) the number of timesteps in the trajectory (can be more than one episode)

    discounts : jax.Array
        Environment discounts `(B, T, 1)`

        Binary values: `1.0` = episode continues, `0.0` = episode ended

        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) the number of timesteps in the trajectory (can be more than one episode)

    values : jax.Array
        State value estimates `(B, T, 1)`.

        - batch_size (`B`) - the number of vectorized environments
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
    preds : PolicyAgentOutput
        Policy network outputs at each step
    target_preds : PolicyAgentOutput
        Target network outputs at each step
    """

    actions: chex.Array
    rewards: chex.Array
    discounts: chex.Array
    values: chex.Array
    preds: PolicyAgentOutput
    target_preds: PolicyAgentOutput

    @property
    def seq_len(self) -> int:
        """Trajectory length (`T`)."""
        return jnp.shape(self.actions)[1]

    @property
    def batch_size(self) -> int:
        """Batch size (`B`)."""
        return jnp.shape(self.actions)[0]

    def squeeze(self) -> Self:
        """
        Removes the last dimension from `(actions, rewards, discounts, values)`
        and returns a new instance of the rollout.

        Returns
        -------
        rollout : Self
            New instance with updates
        """
        return self.__replace__(
            actions=self.actions.squeeze(),
            rewards=self.rewards.squeeze(),
            discounts=self.discounts.squeeze(),
            values=self.values.squeeze(),
        )

    def to_time_first(self) -> Self:
        """
        Transpose rollouts from batch-first to time-first format.

        Converts shape from `(B, T, ...)` to `(T, B, ...)` for all array fields.
        Useful for V-trace and other temporal computations that expect time
        as the leading dimension.

        Returns
        -------
        rollout : Rollout
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


@struct.dataclass
class RolloutStack:
    """
    A stack of `N` rollouts for `jax.lax.scan`.

    Parameters
    ----------
    actions : jax.Array
        Actions taken in the environment `(N, B, T, 1)`

        - n_rollouts (`N`) - the number of rollouts per trajectory
        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)

    rewards : jax.Array
        Rewards generated from the environment `(N, B, T, 1)`

        - n_rollouts (`N`) - the number of rollouts per trajectory
        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)

    discounts : jax.Array
        Environment discounts `(N, B, T, 1)`

        Binary values: `1.0` = episode continues, `0.0` = episode ended

        - n_rollouts (`N`) - the number of rollouts per trajectory
        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)

    values : jax.Array
        State value estimates `(N, B, T, 1)`.

        - n_rollouts (`N`) - the number of rollouts per trajectory
        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)

    preds : PolicyAgentOutput
        A stack of policy network outputs.
        Each field has shape `(N, B, T, ...)`
    target_preds : PolicyAgentOutput
        A stack of target network outputs.
        Each field has shape `(N, B, T, ...)`
    """

    actions: chex.Array
    rewards: chex.Array
    discounts: chex.Array
    values: chex.Array
    preds: PolicyAgentOutput
    target_preds: PolicyAgentOutput

    @property
    def n_rollouts(self) -> int:
        """Number of rollouts (`N`)."""
        return jnp.shape(self.actions)[0]

    def __getitem__(self, idx: int) -> Rollout:
        """
        Gets a single rollout.

        Parameters
        ----------
        idx : int
            Rollout index at `N`

        Returns
        -------
        rollout : Rollout
            The selected rollout
        """
        return Rollout(
            actions=self.actions[idx],  # type: ignore
            rewards=self.rewards[idx],  # type: ignore
            discounts=self.discounts[idx],  # type: ignore
            values=self.values[idx],  # type: ignore
            preds=get_fields_by_index(self.preds, idx),
            target_preds=get_fields_by_index(self.target_preds, idx),
        )

    @classmethod
    def from_list(cls, rollouts: List[Rollout]) -> Self:
        """
        Concatenate a list of rollouts into a stack.

        Parameters
        ----------
        rollouts : List[Rollout]
            `N` rollouts to stack

        Returns
        -------
        stack : RolloutStack
            Stacked rollouts
        """

        def stack_fields(items: List[PolicyAgentOutput]) -> PolicyAgentOutput:
            """Stack fields along new leading dimension."""
            return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *items)

        return cls(
            actions=jnp.stack([r.actions for r in rollouts], axis=0),
            rewards=jnp.stack([r.rewards for r in rollouts], axis=0),
            discounts=jnp.stack([r.discounts for r in rollouts], axis=0),
            values=jnp.stack([r.values for r in rollouts], axis=0),
            preds=stack_fields([r.preds for r in rollouts]),
            target_preds=stack_fields([r.target_preds for r in rollouts]),
        )
