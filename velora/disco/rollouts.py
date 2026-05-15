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

from typing import Self

import jax
import jax.numpy as jnp
from flax import struct

from velora.disco.outputs import PolicyAgentOutput


@struct.dataclass
class Rollout:
    """
    Rollout trajectory data.

    Used in two shapes:

    - Buffer storage: `(P, N, B, T, ...)` — all trainers, all rollouts.
    - Grad chunk (after `get_chunk`): `(C, N, B, T, ...)`, or
      `(C, B, T, ...)` if sliced with `squeeze_n=True`.

    Shape symbols
    -------------
    - `P` / `C` - trainer pool / chunk count.
    - `N` - number of rollouts per collection phase.
    - `B` - batch_size, number of vectorized environments.
    - `T` - sequence length, timesteps in the trajectory.
    - `A` - action dimensionality (max across env set).
    - `E` - encoder output dim.
    - `D` - prediction vector dim (`y` and `z`).

    Parameters
    ----------
    actions : jax.Array
        Actions taken `(..., B, T, A)`.
    rewards : jax.Array
        Environment rewards `(..., B, T, 1)`.
    discounts : jax.Array
        Episode discounts `(..., B, T, 1)`. Binary `1.0` = continues,
        `0.0` = episode ended.
    values : jax.Array
        State value estimates `(..., B, T, 1)`.
    preds : PolicyAgentOutput
        Policy network outputs, each field `(..., B, T, *)`.
    target_preds : PolicyAgentOutput
        Target network outputs, each field `(..., B, T, *)`.
    """

    actions: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    values: jax.Array
    preds: PolicyAgentOutput
    target_preds: PolicyAgentOutput

    def squeeze(self) -> Self:
        """
        Remove the trailing size-1 dimension from `rewards`, `discounts`,
        and `values`. `(..., B, T, 1)` → `(..., B, T)`.

        Returns
        -------
        rollout : Rollout
            New instance with trailing dimension removed.
        """
        return self.__replace__(
            rewards=jnp.squeeze(self.rewards, axis=-1),
            discounts=jnp.squeeze(self.discounts, axis=-1),
            values=jnp.squeeze(self.values, axis=-1),
        )

    def to_time_first(self) -> Self:
        """
        Swap the batch and time axes at positions `1` and `2`.

        For the canonical squeezed chunk shape `(C, B, T, ...)` this
        produces `(C, T, B, ...)` — useful for V-trace and other
        temporal computations that expect time first.

        Returns
        -------
        rollout : Rollout
            New instance with batch/time axes swapped.
        """
        _swap = lambda x: jnp.swapaxes(x, 1, 2)  # noqa: E731
        return self.__replace__(
            actions=_swap(self.actions),
            rewards=_swap(self.rewards),
            discounts=_swap(self.discounts),
            values=_swap(self.values),
            preds=jax.tree.map(_swap, self.preds),
            target_preds=jax.tree.map(_swap, self.target_preds),
        )

    def write_step(
        self,
        rollout_idx: jax.Array,
        step_idx: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        discounts: jax.Array,
        values: jax.Array,
        preds: PolicyAgentOutput,
        target_preds: PolicyAgentOutput,
    ) -> Self:
        """
        Write one timestep across all trainers into rollout/step slot.

        Only valid on the buffer-shaped `(P, N, B, T, ...)` rollout used
        as `PoolRolloutBuffer.as_pytree()`. Returns a new `Rollout` with
        all fields updated via `.at[:, rollout_idx, :, step_idx].set(...)`.
        Both indices are JAX scalars so this composes inside `jax.lax.scan`.

        Parameters
        ----------
        rollout_idx : jax.Array
            `jnp.int32` scalar — rollout slot `n`.
        step_idx : jax.Array
            `jnp.int32` scalar — timestep `t`.
        actions, rewards, discounts, values : jax.Array
            Per-trainer batched values shaped `(P, B, ...)`.
        preds, target_preds : PolicyAgentOutput
            Policy / target predictions shaped `(P, B, ...)`.

        Returns
        -------
        rollout : Rollout
            New `Rollout` with one timestep written.
        """
        n, t = rollout_idx, step_idx
        return self.__replace__(
            actions=self.actions.at[:, n, :, t].set(actions),
            rewards=self.rewards.at[:, n, :, t].set(rewards),
            discounts=self.discounts.at[:, n, :, t].set(discounts),
            values=self.values.at[:, n, :, t].set(values),
            preds=PolicyAgentOutput(
                encoding=self.preds.encoding.at[:, n, :, t].set(preds.encoding),
                mu=self.preds.mu.at[:, n, :, t].set(preds.mu),
                log_std=self.preds.log_std.at[:, n, :, t].set(preds.log_std),
                y=self.preds.y.at[:, n, :, t].set(preds.y),
                z=self.preds.z.at[:, n, :, t].set(preds.z),
                aux_pi=self.preds.aux_pi.at[:, n, :, t].set(preds.aux_pi),
                q=self.preds.q.at[:, n, :, t].set(preds.q),
            ),
            target_preds=PolicyAgentOutput(
                encoding=self.target_preds.encoding.at[:, n, :, t].set(
                    target_preds.encoding
                ),
                mu=self.target_preds.mu.at[:, n, :, t].set(target_preds.mu),
                log_std=self.target_preds.log_std.at[:, n, :, t].set(
                    target_preds.log_std
                ),
                y=self.target_preds.y.at[:, n, :, t].set(target_preds.y),
                z=self.target_preds.z.at[:, n, :, t].set(target_preds.z),
                aux_pi=self.target_preds.aux_pi.at[:, n, :, t].set(target_preds.aux_pi),
                q=self.target_preds.q.at[:, n, :, t].set(target_preds.q),
            ),
        )


class PoolRolloutBuffer:
    """
    Batched rollout buffer for continuous action space trainers.

    Stores all per-trainer experience as a single buffer-shaped
    `Rollout` pytree with leading dims `(P, N, B, T)`. The compiled
    `pool.collect` reads it via `as_pytree`, threads it through the
    `jax.lax.scan` carry, and writes the result back via
    `load_from_pytree`. Grad consumers slice it via `get_chunk`.

    Shape symbols
    -------------
    - `P` - number of trainers.
    - `N` - rollouts per collection phase.
    - `B` - batch_size (vectorized envs per trainer).
    - `T` - timesteps per rollout.
    - `A` - max action dim across env set.
    - `E` - encoder output dim.
    - `D` - prediction vector dim.

    Field layout (each leaf of `self._buffer`):
        - `actions`: `(P, N, B, T, A)`.
        - Policy / target stored as `(mu, log_std)` Gaussian params, each `(P, N, B, T, A)`.
        - `z` has no action dim: `(P, N, B, T, D)`.
        - `aux_pi`: `(P, N, B, T, 2 * A)` — predicted next-step Gaussian params.
        - `q`: scalar `(P, N, B, T, 1)`.

    Parameters
    ----------
    num_trainers : int
        Number of trainers (`P`).
    n_rollouts : int
        Number of rollouts per collection phase (`N`).
    n_envs : int
        Vectorized environments per trainer (`B`).
    seq_len : int
        Timesteps per rollout (`T`).
    n_actions : int
        Max continuous action dim across all environments (`A`).
    encoding_dim : int
        Encoder output dim (`E`).
    prediction_dim : int
        Prediction vector size (`D`) — used for both `y` and `z`.
    dtype : jnp.dtype, optional
        Storage dtype for floating-point fields. Default is `jnp.float32`.
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
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        shape = (num_trainers, n_rollouts, n_envs, seq_len)  # (P, N, B, T)
        zeros = lambda last: jnp.zeros((*shape, last), dtype=dtype)  # noqa: E731

        def _preds() -> PolicyAgentOutput:
            return PolicyAgentOutput(
                encoding=zeros(encoding_dim),
                mu=zeros(n_actions),
                log_std=zeros(n_actions),
                y=zeros(prediction_dim),
                z=zeros(prediction_dim),
                aux_pi=zeros(2 * n_actions),
                q=zeros(1),
            )

        self._buffer = Rollout(
            actions=zeros(n_actions),
            rewards=zeros(1),
            discounts=zeros(1),
            values=zeros(1),
            preds=_preds(),
            target_preds=_preds(),
        )

    def as_pytree(self) -> Rollout:
        """
        Return the buffer-shaped `Rollout` for scan carry.

        Returns
        -------
        rollout : Rollout
            The full `(P, N, B, T, ...)` buffer pytree (zero-copy).
        """
        return self._buffer

    def load_from_pytree(self, rollout: Rollout) -> None:
        """
        Write back a post-scan buffer-shaped `Rollout`.

        Parameters
        ----------
        rollout : Rollout
            Buffer-shaped `Rollout` produced by the scan's final carry.
        """
        self._buffer = rollout

    def get_chunk(self, start: int, end: int, squeeze_n: bool = False) -> Rollout:
        """
        Slice trainers `[start:end]` as a `Rollout`. All arrays are
        already on device — this is a zero-copy view.

        Parameters
        ----------
        start : int
            First trainer index (inclusive).
        end : int
            Last trainer index (exclusive).
        squeeze_n : bool, optional
            Remove the rollout dimension `N`. Default is `False`.

        Returns
        -------
        rollout : Rollout
            Stacked rollout with `PolicyAgentOutput` preds.
        """
        rollout = jax.tree.map(lambda x: x[start:end], self._buffer)
        if squeeze_n:
            rollout = jax.tree.map(lambda x: x[:, 0], rollout)
        return rollout

    def memory_mb(self) -> float:
        """
        Total size of all pre-allocated buffers in MB.

        Returns
        -------
        size_mb : float
            Combined size of all JAX buffers in megabytes.
        """
        return sum(a.nbytes for a in jax.tree.leaves(self._buffer)) / 1e6

    def __repr__(self) -> str:
        P, N, B, T = jnp.shape(self._buffer.actions)[:4]
        D = jnp.shape(self._buffer.actions)[4]

        return (
            f"PoolRolloutBuffer("
            f"trainers={P}, n_rollouts={N}, n_envs={B}, seq_len={T}, "
            f"max_action_dim={D}, "
            f"memory={self.memory_mb():.1f}MB"
            f")"
        )
