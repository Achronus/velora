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

from typing import TYPE_CHECKING, List, Self

import chex
import jax
import jax.numpy as jnp
from flax import struct

if TYPE_CHECKING:
    from velora.disco.outputs import PolicyAgentOutput

from velora.utils.structs import get_fields_by_index
from velora.utils.transforms import to_time_first


@struct.dataclass
class Rollout:
    """
    A single `(B, T, 1)` or stack of `N` rollouts `(N, B, T, 1)`.

    Parameters
    ----------
    actions : jax.Array
        Actions taken in the environment `(N, B, T, 1)` or `(B, T, 1)`

        - n_rollouts (`N`) - the number of rollouts per trajectory
        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) the number of timesteps in the trajectory (can be more than one episode)

    rewards : jax.Array
        Rewards generated from the environment `(N, B, T, 1)` or `(B, T, 1)`

        - n_rollouts (`N`) - the number of rollouts per trajectory
        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) the number of timesteps in the trajectory (can be more than one episode)

    discounts : jax.Array
        Environment discounts `(N, B, T, 1)` or `(B, T, 1)`

        Binary values: `1.0` = episode continues, `0.0` = episode ended

        - n_rollouts (`N`) - the number of rollouts per trajectory
        - batch_size (`B`) the number of vectorized environments
        - seq_length (`T`) the number of timesteps in the trajectory (can be more than one episode)

    values : jax.Array
        State value estimates `(N, B, T, 1)` or `(B, T, 1)`.

        - n_rollouts (`N`) - the number of rollouts per trajectory
        - batch_size (`B`) - the number of vectorized environments
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
    preds : PolicyAgentOutput
        Policy network outputs.
        Each field has shape `(N, B, T, ...)` or `(B, T, ...)`
    target_preds : PolicyAgentOutput
        Target network outputs.
        Each field has shape `(N, B, T, ...)` or `(B, T, ...)`
    """

    actions: chex.Array
    rewards: chex.Array
    discounts: chex.Array
    values: chex.Array
    preds: "PolicyAgentOutput"
    target_preds: "PolicyAgentOutput"

    @property
    def is_stacked(self) -> bool:
        """True if shape `(N, B, T, ...)`."""
        return jnp.ndim(self.actions) == 4

    @property
    def seq_len(self) -> int:
        """Trajectory length (`T`)."""
        if self.is_stacked:
            return jnp.shape(self.actions)[2]

        return jnp.shape(self.actions)[1]

    @property
    def batch_size(self) -> int:
        """Batch size (`B`)."""
        if self.is_stacked:
            return jnp.shape(self.actions)[1]

        return jnp.shape(self.actions)[0]

    @property
    def n_rollouts(self) -> int:
        """Number of rollouts (`N`)."""
        if self.is_stacked:
            return jnp.shape(self.actions)[0]

        return 1

    def squeeze(self) -> Self:
        """
        Removes size-1 dimension from `(actions, rewards, discounts, values)`
        and returns a new instance of the rollout.

        Handles both shapes:
            - Unstacked `(B, T, 1)`    → `(B, T)`
            - Stacked   `(N, B, T, 1)` → `(N, B, T)`

        Returns
        -------
        rollout : Rollout
            New instance with trailing dimension removed
        """
        return self.__replace__(
            actions=jnp.squeeze(self.actions, axis=-1),
            rewards=jnp.squeeze(self.rewards, axis=-1),
            discounts=jnp.squeeze(self.discounts, axis=-1),
            values=jnp.squeeze(self.values, axis=-1),
        )

    def to_time_first(self) -> Self:
        """
        Transpose from batch-first to time-first format.

        Handles both shapes:
            - Unstacked `(B, T, ...)`    → `(T, B, ...)`
            - Stacked   `(N, B, T, ...)` → `(N, T, B, ...)` — transposes axes 1 and 2

        Useful for V-trace and other temporal computations that expect time
        as the leading dimension.

        Returns
        -------
        rollout : Rollout
            New instance with time-first arrays
        """
        _swap = (lambda x: jnp.swapaxes(x, 1, 2)) if self.is_stacked else to_time_first

        return self.__replace__(
            actions=_swap(self.actions),
            rewards=_swap(self.rewards),
            discounts=_swap(self.discounts),
            values=_swap(self.values),
            preds=jax.tree.map(_swap, self.preds),
            target_preds=jax.tree.map(_swap, self.target_preds),
        )

    def __getitem__(self, idx: int) -> Self:
        """
        Extract a single unstacked rollout by index.

        Parameters
        ----------
        idx : int
            Rollout index at `N` axis

        Returns
        -------
        rollout : Rollout
            A single `(B, T, ...)` rollout

        Raises
        ------
        unstacked: IndexError
            If called on an unstacked rollout
        """
        if not self.is_stacked:
            raise IndexError(
                "Rollout is not stacked. Use `from_list()` to create a stacked "
                "rollout before indexing."
            )

        return Rollout(
            actions=self.actions[idx],  # type: ignore
            rewards=self.rewards[idx],  # type: ignore
            discounts=self.discounts[idx],  # type: ignore
            values=self.values[idx],  # type: ignore
            preds=get_fields_by_index(self.preds, idx),
            target_preds=get_fields_by_index(self.target_preds, idx),
        )

    @classmethod
    def from_list(cls, rollouts: List[Self]) -> Self:
        """
        Stack a list of rollouts into a single `(N, B, T, ...)` rollout.

        Parameters
        ----------
        rollouts : List[Rollout]
            `N` unstacked rollouts to stack

        Returns
        -------
        stack : Rollout
            Stacked rollout with `(N, B, T, ...)`
        """

        def stack_fields(items: List["PolicyAgentOutput"]) -> "PolicyAgentOutput":
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
