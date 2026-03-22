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
import numpy as np
from flax import struct

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
    preds: PolicyAgentOutput
    target_preds: PolicyAgentOutput

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


class PoolRolloutBuffer:
    """
    Batched rollout buffer for all trainers in a `TrainerPool`.

    Stores experience for all trainers in a single pre-allocated numpy
    array with shape `(P, N, B, T, ...)`, where `P` is the number of
    trainers.

    Supports batched writes (one `write_step_batched` call per
    collection step for all trainers simultaneously) and chunk-based
    reads (`get_chunk` slices and transfers a trainer range directly
    to GPU for the vmapped gradient function).

    Parameters
    ----------
    num_trainers : int
        Number of trainers in the pool (`P`)
    n_rollouts : int
        Number of rollouts per collection phase (`N`).
        Training uses `n_updates`, validation uses `1`
    n_envs : int
        Number of vectorized environments per trainer (`B`)
    seq_len : int
        Number of timesteps per rollout (`T`)
    n_actions : int
        Maximum action space size across all environments
    encoding_dim : int
        Dimensionality of the CNN encoder output
    prediction_dim : int
        Dimensionality of `y` and `z` prediction vectors
    q_dim : int
        Number of bins in the distributional Q-value head
    use_bfloat16 : bool (optional)
        Cast floating-point arrays to `bfloat16` on GPU transfer.
        Default is `True`
    """

    def __init__(
        self,
        num_trainers: int,
        n_rollouts: int,
        n_envs: int,
        seq_len: int,
        n_actions: int,
        encoding_dim: int,
        prediction_dim: int,
        q_dim: int,
        *,
        use_bfloat16: bool = True,
    ) -> None:
        self._num_trainers = num_trainers
        self._n_rollouts = n_rollouts
        self._use_bfloat16 = use_bfloat16

        shape = (num_trainers, n_rollouts, n_envs, seq_len)  # (P, N, B, T)

        # Core arrays
        self.actions = np.zeros((*shape, 1), dtype=np.int32)
        self.rewards = np.zeros((*shape, 1), dtype=np.float32)
        self.discounts = np.zeros((*shape, 1), dtype=np.float32)
        self.values = np.zeros((*shape, 1), dtype=np.float32)

        # Policy prediction arrays
        self.p_encoding = np.zeros((*shape, encoding_dim), dtype=np.float32)
        self.p_pi = np.zeros((*shape, n_actions), dtype=np.float32)
        self.p_y = np.zeros((*shape, prediction_dim), dtype=np.float32)
        self.p_z = np.zeros((*shape, n_actions, prediction_dim), dtype=np.float32)
        self.p_aux_pi = np.zeros((*shape, n_actions, n_actions), dtype=np.float32)
        self.p_q = np.zeros((*shape, n_actions, q_dim), dtype=np.float32)

        # Target prediction arrays
        self.t_encoding = np.zeros((*shape, encoding_dim), dtype=np.float32)
        self.t_pi = np.zeros((*shape, n_actions), dtype=np.float32)
        self.t_y = np.zeros((*shape, prediction_dim), dtype=np.float32)
        self.t_z = np.zeros((*shape, n_actions, prediction_dim), dtype=np.float32)
        self.t_aux_pi = np.zeros((*shape, n_actions, n_actions), dtype=np.float32)
        self.t_q = np.zeros((*shape, n_actions, q_dim), dtype=np.float32)

        # Write indices
        self._rollout_idx = 0
        self._step_idx = 0

    def write_step_batched(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        discounts: np.ndarray,
        values: np.ndarray,
        preds: PolicyAgentOutput,
        target_preds: PolicyAgentOutput,
    ) -> None:
        """
        Write one timestep for ALL trainers simultaneously.

        All inputs have a leading `(P, ...)` dimension. Uses
        `np.copyto` into pre-allocated memory — zero allocations.

        Parameters
        ----------
        actions : np.ndarray
            `(P, B, 1)` int32
        rewards : np.ndarray
            `(P, B, 1)` float32
        discounts : np.ndarray
            `(P, B, 1)` float32
        values : np.ndarray
            `(P, B, 1)` float32
        preds : PolicyAgentOutput
            Policy predictions with fields `(P, B, ...)` as numpy
        target_preds : PolicyAgentOutput
            Target predictions with fields `(P, B, ...)` as numpy
        """
        n = self._rollout_idx
        t = self._step_idx

        # Core
        np.copyto(self.actions[:, n, :, t], actions)
        np.copyto(self.rewards[:, n, :, t], rewards)
        np.copyto(self.discounts[:, n, :, t], discounts)
        np.copyto(self.values[:, n, :, t], values)

        # Policy preds
        np.copyto(self.p_encoding[:, n, :, t], preds.encoding)
        np.copyto(self.p_pi[:, n, :, t], preds.pi)
        np.copyto(self.p_y[:, n, :, t], preds.y)
        np.copyto(self.p_z[:, n, :, t], preds.z)
        np.copyto(self.p_aux_pi[:, n, :, t], preds.aux_pi)
        np.copyto(self.p_q[:, n, :, t], preds.q)

        # Target preds
        np.copyto(self.t_encoding[:, n, :, t], target_preds.encoding)
        np.copyto(self.t_pi[:, n, :, t], target_preds.pi)
        np.copyto(self.t_y[:, n, :, t], target_preds.y)
        np.copyto(self.t_z[:, n, :, t], target_preds.z)
        np.copyto(self.t_aux_pi[:, n, :, t], target_preds.aux_pi)
        np.copyto(self.t_q[:, n, :, t], target_preds.q)

        self._step_idx += 1

    def next_rollout(self) -> None:
        """Advance all trainers to the next rollout slot."""
        self._rollout_idx += 1
        self._step_idx = 0

    def reset_head(self) -> None:
        """Reset write indices for the next collection phase."""
        self._rollout_idx = 0
        self._step_idx = 0

    def get_chunk(self, start: int, end: int, squeeze_n: bool = False) -> Rollout:
        """
        Slice trainers `[start:end]` and transfer to accelerator as a `Rollout`.

        Directly creates stacked JAX arrays from the numpy buffer — no
        intermediate `stack_pytrees` call needed. Applies `bfloat16`
        conversion if enabled.

        Parameters
        ----------
        start : int
            First trainer index (inclusive)
        end : int
            Last trainer index (exclusive)
        squeeze_n : bool (optional)
            Remove the rollout dimension `N` from the output, converting
            `(C, N, B, T, ...)` to `(C, B, T, ...)`. Used for
            validation buffers where `N=1`. Default is `False`

        Returns
        -------
        rollout : Rollout
            Stacked rollout `(chunk_size, N, B, T, ...)` on GPU
        """
        s = slice(start, end)

        rollout = Rollout(
            actions=jnp.asarray(self.actions[s]),
            rewards=jnp.asarray(self.rewards[s]),
            discounts=jnp.asarray(self.discounts[s]),
            values=jnp.asarray(self.values[s]),
            preds=PolicyAgentOutput(
                encoding=jnp.asarray(self.p_encoding[s]),
                pi=jnp.asarray(self.p_pi[s]),
                y=jnp.asarray(self.p_y[s]),
                z=jnp.asarray(self.p_z[s]),
                aux_pi=jnp.asarray(self.p_aux_pi[s]),
                q=jnp.asarray(self.p_q[s]),
            ),
            target_preds=PolicyAgentOutput(
                encoding=jnp.asarray(self.t_encoding[s]),
                pi=jnp.asarray(self.t_pi[s]),
                y=jnp.asarray(self.t_y[s]),
                z=jnp.asarray(self.t_z[s]),
                aux_pi=jnp.asarray(self.t_aux_pi[s]),
                q=jnp.asarray(self.t_q[s]),
            ),
        )

        if self._use_bfloat16:
            rollout = jax.tree.map(
                lambda x: (
                    x.astype(jnp.bfloat16)
                    if jnp.issubdtype(x.dtype, jnp.floating)
                    else x
                ),
                rollout,
            )

        if squeeze_n:
            rollout = jax.tree.map(lambda x: x[:, 0], rollout)

        return rollout

    def get_single(self, trainer_idx: int) -> Rollout:
        """
        Get a single trainer's rollout and transfer to accelerator.

        Parameters
        ----------
        trainer_idx : int
            Trainer index

        Returns
        -------
        rollout : Rollout
            Single trainer rollout `(N, B, T, ...)` on accelerator
        """
        return self.get_chunk(trainer_idx, trainer_idx + 1)

    def memory_mb(self) -> float:
        """
        Total size of all pre-allocated numpy arrays in MB.

        Returns
        -------
        size_mb : float
            Combined size of all numpy buffers in megabytes
        """
        arrays = [
            self.actions,
            self.rewards,
            self.discounts,
            self.values,
            self.p_encoding,
            self.p_pi,
            self.p_y,
            self.p_z,
            self.p_aux_pi,
            self.p_q,
            self.t_encoding,
            self.t_pi,
            self.t_y,
            self.t_z,
            self.t_aux_pi,
            self.t_q,
        ]
        return sum(a.nbytes for a in arrays) / 1e6

    def __repr__(self) -> str:
        P, N, B, T = self.actions.shape[:4]

        return (
            f"MegaRolloutBuffer("
            f"trainers={P}, n_rollouts={N}, n_envs={B}, seq_len={T}, "
            f"rollout={self._rollout_idx}/{N}, step={self._step_idx}, "
            f"memory={self.memory_mb():.1f}MB"
            f")"
        )
