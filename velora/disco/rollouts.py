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


class RolloutBuffer:
    """
    Pre-allocated CPU buffer for a stack of `N` rollouts.

    All step data is written into fixed-size numpy arrays on the CPU.
    A single `to_rollout()` call transfers the completed buffer to JAX
    in one operation.

    The buffer is circular and reused across meta-steps.

    Parameters
    ----------
    n_rollouts : int
        Number of rollouts to collect per `collect_stack` call (`N`).
        Corresponds to `config.n_updates`.
    n_envs : int
        Number of parallel vectorised environments (`B`).
        Corresponds to `config.num_vec_envs`.
    seq_len : int
        Trajectory length per rollout in timesteps (`T`).
        Corresponds to `config.seq_len`.
    n_actions : int
        Size of the discrete action space. Used to allocate `pi`,
        `aux_pi`, and `q` arrays.
    encoding_dim : int
        Dimensionality of the CNN encoding output. Shared between
        `preds` and `target_preds`.
    prediction_dim : int
        Dimensionality of the observation-conditioned prediction vector
        `y` and the action-conditioned prediction vector `z`.
    q_dim : int
        Number of bins in the distributional action-value head `q`.
        Set to `n_actions` for a standard scalar Q head.

    Attributes
    ----------
    actions : np.ndarray
        int32 `(N, B, T, 1)` — Actions taken at each step.
    rewards : np.ndarray
        float32 `(N, B, T, 1)` — Rewards received at each step.
    discounts : np.ndarray
        float32 `(N, B, T, 1)` — Episode continuation mask.
        `1.0` = episode continues, `0.0` = episode ended.
    values : np.ndarray
        float32 `(N, B, T, 1)` — State-value estimates.

    Policy prediction fields (prefix `p_`):
        p_encoding : np.ndarray  float32 `(N, B, T, encoding_dim)`
        p_pi       : np.ndarray  float32 `(N, B, T, n_actions)`
        p_y        : np.ndarray  float32 `(N, B, T, prediction_dim)`
        p_z        : np.ndarray  float32 `(N, B, T, prediction_dim)`
        p_aux_pi   : np.ndarray  float32 `(N, B, T, n_actions, n_actions)`
        p_q        : np.ndarray  float32 `(N, B, T, n_actions, q_dim)`

    Target prediction fields (prefix `t_`):
        t_encoding : np.ndarray  float32 `(N, B, T, encoding_dim)`
        t_pi       : np.ndarray  float32 `(N, B, T, n_actions)`
        t_y        : np.ndarray  float32 `(N, B, T, prediction_dim)`
        t_z        : np.ndarray  float32 `(N, B, T, prediction_dim)`
        t_aux_pi   : np.ndarray  float32 `(N, B, T, n_actions, n_actions)`
        t_q        : np.ndarray  float32 `(N, B, T, n_actions, q_dim)`
    """

    def __init__(
        self,
        n_rollouts: int,
        n_envs: int,
        seq_len: int,
        n_actions: int,
        encoding_dim: int,
        prediction_dim: int,
        q_dim: int,
    ):
        shape = (n_rollouts, n_envs, seq_len)  # (N, B, T)

        # Core
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

        # Indexing
        self._n_rollouts = n_rollouts
        self._rollout_idx = 0
        self._step_idx = 0

    def write_step(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        discounts: np.ndarray,
        values: np.ndarray,
        preds: PolicyAgentOutput,
        target_preds: PolicyAgentOutput,
    ):
        """
        Write a single environment step into the buffer.

        Uses `np.copyto` throughout to write into pre-allocated memory
        without creating any new numpy or JAX arrays.

        Parameters
        ----------
        actions : np.ndarray
            int32 `(B, 1)` — Actions taken by the policy.
        rewards : np.ndarray
            float32 `(B, 1)` — Rewards returned by the environment.
        discounts : np.ndarray
            float32 `(B, 1)` — Episode continuation mask from the
            environment. `1.0` = continues, `0.0` = ended.
        values : np.ndarray
            float32 `(B, 1)` — State-value estimates from the value
            network.
        preds : PolicyAgentOutput
            Predictions from the live policy agent at this step.
            Expected fields: `encoding`, `pi`, `y`, `z`,
            `aux_pi`, `q`.
        target_preds : PolicyAgentOutput
            Predictions from the target agent at this step.
            Same fields as `preds`.

        Raises
        ------
        IndexError
            If called after the buffer is full (i.e., after
            ``n_rollouts`` calls to ``next_rollout()``).
        """
        n = self._rollout_idx
        t = self._step_idx

        if n >= self._n_rollouts:
            raise IndexError(
                f"Buffer is full. Called write_step() after {self._n_rollouts} "
                "rollouts. Call to_rollout() and recreate the buffer."
            )

        p = preds.to_numpy()
        tp = target_preds.to_numpy()

        # Core
        np.copyto(self.actions[n, :, t], actions)
        np.copyto(self.rewards[n, :, t], rewards)
        np.copyto(self.discounts[n, :, t], discounts)
        np.copyto(self.values[n, :, t], values)

        # Policy preds
        np.copyto(self.p_encoding[n, :, t], p.encoding)
        np.copyto(self.p_pi[n, :, t], p.pi)
        np.copyto(self.p_y[n, :, t], p.y)
        np.copyto(self.p_z[n, :, t], p.z)
        np.copyto(self.p_aux_pi[n, :, t], p.aux_pi)
        np.copyto(self.p_q[n, :, t], p.q)

        # Target preds
        np.copyto(self.t_encoding[n, :, t], tp.encoding)
        np.copyto(self.t_pi[n, :, t], tp.pi)
        np.copyto(self.t_y[n, :, t], tp.y)
        np.copyto(self.t_z[n, :, t], tp.z)
        np.copyto(self.t_aux_pi[n, :, t], tp.aux_pi)
        np.copyto(self.t_q[n, :, t], tp.q)

        self._step_idx += 1

    def next_rollout(self):
        """
        Advance the write-head to the next rollout slot.

        Resets the step index to zero and increments the rollout index.
        Must be called once after every `seq_len` call to
        `write_step()`.

        Note
        ----
        This does **not** clear the buffer. Old data in the next slot
        will be overwritten naturally by subsequent `write_step()`
        calls before `to_rollout()` reads it.
        """
        self._rollout_idx += 1
        self._step_idx = 0

    def to_rollout(self) -> Rollout:
        """
        Transfer the completed buffer to JAX and return a stacked
        `Rollout`.

        Returns
        -------
        rollout : Rollout
            Stacked rollout with shape `(N, B, T, ...)` on the default
            JAX device (GPU when available, CPU otherwise).

        Note
        ----
        The buffer's numpy arrays are not freed after this call — they
        are reused on the next iteration. This is intentional: the
        allocation cost is paid once at construction time.
        """
        rollout = Rollout(
            actions=jnp.asarray(self.actions),
            rewards=jnp.asarray(self.rewards),
            discounts=jnp.asarray(self.discounts),
            values=jnp.asarray(self.values),
            preds=PolicyAgentOutput.from_numpy(
                self.p_encoding,
                self.p_pi,
                self.p_y,
                self.p_z,
                self.p_aux_pi,
                self.p_q,
            ),
            target_preds=PolicyAgentOutput.from_numpy(
                self.t_encoding,
                self.t_pi,
                self.t_y,
                self.t_z,
                self.t_aux_pi,
                self.t_q,
            ),
        )

        # Reset write-head for the next collect_stack call
        self._rollout_idx = 0
        self._step_idx = 0

        return rollout

    def memory_mb(self) -> float:
        """
        Return the total size of all pre-allocated numpy arrays in MB.

        Useful for logging at construction time to confirm the buffer
        footprint before training begins.

        Returns
        -------
        size_mb : float
            Combined size of all numpy buffers in megabytes.
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
        n, b, t = (
            self._n_rollouts,
            self.actions.shape[1],
            self.actions.shape[2],
        )
        return (
            f"RolloutBuffer("
            f"n_rollouts={n}, n_envs={b}, seq_len={t}, "
            f"rollout={self._rollout_idx}/{n}, step={self._step_idx}, "
            f"memory={self.memory_mb():.1f}MB"
            f")"
        )
