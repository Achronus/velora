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

from typing import Self, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from velora.disco.outputs import PolicyAgentOutput


@struct.dataclass
class Rollout:
    """
    Rollout trajectory data.

    Used in two shapes:

    - Training storage (inside `MixedBuffer.storage`): `(P, capacity, T, ...)`
      — per-agent ring of trajectories.
    - Validation storage (raw `Rollout` on the pool): `(P, 1, 2T, ...)`
      — one fresh on-policy rollout per meta-step.

    Grad chunks consume:
    - Training: `(C, N, B, T, ...)` from `MixedBuffer.sample` —
      `B = batch_size` mixed fresh+replay trajectories per agent update.
    - Validation: `(C, 1, 2T, ...)` after slicing trainers and inserting
      a unit `B=1` axis at the call site.

    Shape symbols
    -------------
    - `P` / `C` - trainer pool / chunk count.
    - `N` - number of agent updates per meta-step.
    - `T` - sequence length, timesteps in the trajectory.
    - `A` - action dimensionality (max across env set).
    - `E` - encoder output dim.
    - `D` - prediction vector dim (`y` and `z`).

    Parameters
    ----------
    actions : jax.Array
        Actions taken `(..., T, A)`.
    rewards : jax.Array
        Environment rewards `(..., T, 1)`.
    discounts : jax.Array
        Episode discounts `(..., T, 1)`. Binary `1.0` = continues,
        `0.0` = episode ended.
    values : jax.Array
        State value estimates `(..., T, 1)`.
    preds : PolicyAgentOutput
        Policy network outputs, each field `(..., T, *)`.
    target_preds : PolicyAgentOutput
        Target network outputs, each field `(..., T, *)`.
    """

    actions: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    values: jax.Array
    preds: PolicyAgentOutput
    target_preds: PolicyAgentOutput

    @classmethod
    def zeros(
        cls,
        shape_prefix: Tuple[int, ...],
        n_actions: int,
        encoding_dim: int,
        prediction_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> Self:
        """
        Allocate a zero-initialised `Rollout` with leading `shape_prefix`.

        Parameters
        ----------
        shape_prefix : Tuple[int, ...]
            Leading shape applied before each leaf's trailing feature
            dim, e.g. `(P, N, T)` for the standard buffer layout.
        n_actions : int
            Max action dim across all environments (`A`).
        encoding_dim : int
            Encoder output dim (`E`).
        prediction_dim : int
            Prediction vector size (`D`) — used for `y` and `z` vectors.
        dtype : jnp.dtype, optional
            Storage dtype for floating-point fields. Default is `jnp.float32`.

        Returns
        -------
        rollout : Rollout
            Zero-initialised rollout pytree.
        """
        zeros = lambda last: jnp.zeros((*shape_prefix, last), dtype=dtype)  # noqa: E731

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

        return cls(
            actions=zeros(n_actions),
            rewards=zeros(1),
            discounts=zeros(1),
            values=zeros(1),
            preds=_preds(),
            target_preds=_preds(),
        )

    def squeeze(self) -> Self:
        """
        Remove the trailing size-1 dimension from `rewards`, `discounts`,
        and `values`. `(..., T, 1)` → `(..., T)`.

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
        Swap the batch and time axes (`B` and `T`).

        Uses negative indices (`-3`, `-2`) so the swap works whether the
        rollout has a leading chunk dim (`(C, B, T, ...)` outside vmap)
        or has been collapsed by vmap (`(B, T, ...)`). All `Rollout`
        fields share the `(..., B, T, *)` layout so the swap is uniform.

        Returns
        -------
        rollout : Rollout
            New instance with batch/time axes swapped.
        """
        _swap = lambda x: jnp.swapaxes(x, -3, -2)  # noqa: E731
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
        slot_idx: jax.Array,
        step_idx: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        discounts: jax.Array,
        values: jax.Array,
        preds: PolicyAgentOutput,
        target_preds: PolicyAgentOutput,
    ) -> Self:
        """
        Write one timestep across all trainers into a buffer slot.

        Only valid on buffer-shaped `(P, S, T, ...)` rollouts where `S`
        is the slot axis (e.g. `capacity` for `MixedBuffer.storage`, `1`
        for the validation rollout). Returns a new `Rollout` with all
        fields updated via `.at[:, slot_idx, step_idx].set(...)`. Both
        indices are JAX scalars so this composes inside `jax.lax.scan`.

        Parameters
        ----------
        slot_idx : jax.Array
            `jnp.int32` scalar — slot position along the `S` axis.
        step_idx : jax.Array
            `jnp.int32` scalar — timestep `t`.
        actions, rewards, discounts, values : jax.Array
            Per-trainer values shaped `(P, ...)`.
        preds, target_preds : PolicyAgentOutput
            Policy / target predictions shaped `(P, ...)`.

        Returns
        -------
        rollout : Rollout
            New `Rollout` with one timestep written.
        """
        n, t = slot_idx, step_idx
        return self.__replace__(
            actions=self.actions.at[:, n, t].set(actions),
            rewards=self.rewards.at[:, n, t].set(rewards),
            discounts=self.discounts.at[:, n, t].set(discounts),
            values=self.values.at[:, n, t].set(values),
            preds=PolicyAgentOutput(
                encoding=self.preds.encoding.at[:, n, t].set(preds.encoding),
                mu=self.preds.mu.at[:, n, t].set(preds.mu),
                log_std=self.preds.log_std.at[:, n, t].set(preds.log_std),
                y=self.preds.y.at[:, n, t].set(preds.y),
                z=self.preds.z.at[:, n, t].set(preds.z),
                aux_pi=self.preds.aux_pi.at[:, n, t].set(preds.aux_pi),
                q=self.preds.q.at[:, n, t].set(preds.q),
            ),
            target_preds=PolicyAgentOutput(
                encoding=self.target_preds.encoding.at[:, n, t].set(
                    target_preds.encoding
                ),
                mu=self.target_preds.mu.at[:, n, t].set(target_preds.mu),
                log_std=self.target_preds.log_std.at[:, n, t].set(target_preds.log_std),
                y=self.target_preds.y.at[:, n, t].set(target_preds.y),
                z=self.target_preds.z.at[:, n, t].set(target_preds.z),
                aux_pi=self.target_preds.aux_pi.at[:, n, t].set(target_preds.aux_pi),
                q=self.target_preds.q.at[:, n, t].set(target_preds.q),
            ),
        )


