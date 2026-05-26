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

from functools import partial
from typing import Self

import chex
import jax
import jax.numpy as jnp
from flax import struct

from velora.disco.outputs import PolicyAgentOutput
from velora.disco.rollouts import Rollout


@partial(
    jax.jit,
    static_argnames=(
        "num_trainers",
        "n_updates",
        "n_fresh_per_update",
        "n_replay_per_update",
        "n_fresh_per_meta",
        "capacity",
    ),
)
def _sample_compiled(
    storage: Rollout,
    write_idx: jax.Array,
    valid_count: jax.Array,
    rng: chex.PRNGKey,
    num_trainers: int,
    n_updates: int,
    n_fresh_per_update: int,
    n_replay_per_update: int,
    n_fresh_per_meta: int,
    capacity: int,
) -> Rollout:
    """
    JIT'd body of `MixedBuffer.sample`. Lifted to module scope so the
    function identity is stable across calls — JAX caches the compile
    once instead of re-tracing every meta-step.

    Vmaps over the full leading `num_trainers` axis in one program;
    chunking was removed in the Phase 2 throughput refactor.
    """
    fresh_offsets = jnp.arange(n_fresh_per_meta)
    fresh_indices = (write_idx - n_fresh_per_meta + fresh_offsets) % capacity
    fresh_indices = fresh_indices.reshape(n_updates, n_fresh_per_update)

    rngs = jax.random.split(rng, num_trainers)

    def _sample_one(
        agent_storage: Rollout,
        v_count: jax.Array,
        agent_rng: chex.PRNGKey,
    ) -> Rollout:
        replay_maxval = jnp.maximum(v_count, 1)
        replay_indices = jax.random.randint(
            agent_rng,
            (n_updates, n_replay_per_update),
            0,
            replay_maxval,
        )
        all_indices = jnp.concatenate([fresh_indices, replay_indices], axis=1)
        return jax.tree.map(lambda x: x[all_indices], agent_storage)

    return jax.vmap(_sample_one)(storage, valid_count, rngs)


@struct.dataclass
class MixedBuffer:
    """
    Per-agent ring buffer of trajectories supporting the DiscoRL
    fresh + replay sampling scheme.

    Uses a deterministic slice of the most recently inserted (fresh)
    trajectories alongside a uniform-random draw from the populated
    buffer (replay).

    Shape symbols
    -------------
    - `P` - total trainer count.
    - `capacity` - per-agent ring slots.
    - `T` - timesteps per trajectory.

    Parameters
    ----------
    storage : Rollout
        Trajectory storage, each leaf shaped `(P, capacity, T, *)`.
    write_idx : jax.Array
        Scalar ring write cursor `int32`. Shared across all agents
        (collection advances them in lockstep).
    valid_count : jax.Array
        Per-agent populated slot count `(P,)` `int32`, capped at
        `capacity`. Per-agent so individual trainer resets can clear
        one agent's history without disturbing the others.
    capacity : int
        Per-agent ring size.
    replay_ratio : float
        Fraction of each per-update batch drawn from replay; the rest is
        taken from the most-recently inserted fresh trajectories.
    """

    storage: Rollout
    write_idx: jax.Array
    valid_count: jax.Array
    capacity: int = struct.field(pytree_node=False)
    replay_ratio: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        num_trainers: int,
        capacity: int,
        replay_ratio: float,
        seq_len: int,
        n_actions: int,
        encoding_dim: int,
        prediction_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> Self:
        """
        Allocate an empty `MixedBuffer`.

        Parameters
        ----------
        num_trainers : int
            Number of trainers (`P`).
        capacity : int
            Per-agent ring size.
        replay_ratio : float
            Fraction of each per-update batch drawn from replay. Must be
            in `[0.0, 1.0]`.
        seq_len : int
            Timesteps per trajectory (`T`).
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
        buffer : MixedBuffer
            Empty buffer with all `write_idx` and `valid_count` at zero.
        """
        if not 0.0 <= replay_ratio <= 1.0:
            raise ValueError(f"replay_ratio must be in [0.0, 1.0], got {replay_ratio}")

        shape = (num_trainers, capacity, seq_len)  # (P, capacity, T)
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

        rollout = Rollout(
            actions=zeros(n_actions),
            rewards=zeros(1),
            discounts=zeros(1),
            values=zeros(1),
            preds=_preds(),
            target_preds=_preds(),
        )

        return cls(
            storage=rollout,
            write_idx=jnp.zeros((), dtype=jnp.int32),
            valid_count=jnp.zeros((num_trainers,), dtype=jnp.int32),
            capacity=capacity,
            replay_ratio=replay_ratio,
        )

    def add(self, fresh: Rollout) -> Self:
        """
        Insert a batch of new trajectories into the buffer.

        Automatically overwrites oldest entries when `capacity` is reached.

        Parameters
        ----------
        fresh : Rollout
            Fresh trajectories shaped `(P, n_fresh, T, *)` per leaf.

        Returns
        -------
        new_buffer : MixedBuffer
            Updated buffer with `n_fresh` trajectories added per agent.
        """
        n_fresh = jnp.shape(fresh.actions)[1]
        capacity = self.capacity
        indices = (self.write_idx + jnp.arange(n_fresh)) % capacity  # (n_fresh,)

        # Same indices for every agent; scatter along axis 1 of each leaf.
        new_storage = jax.tree.map(
            lambda dst, src: dst.at[:, indices].set(src),
            self.storage,
            fresh,
        )
        new_write_idx = (self.write_idx + n_fresh) % capacity
        new_valid_count = jnp.minimum(self.valid_count + n_fresh, capacity)

        return self.__replace__(
            storage=new_storage,
            write_idx=new_write_idx,
            valid_count=new_valid_count,
        )

    def sample(
        self,
        rng: chex.PRNGKey,
        n_updates: int,
        batch_size: int,
    ) -> Rollout:
        """
        Build a `(P, n_updates, batch_size, T, *)` mixed batch covering
        every agent in one vmapped pass.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key — split per agent for independent replay draws.
        n_updates : int
            Number of per-update batches to produce (`N`).
        batch_size : int
            Trajectories per per-update batch (`B`).

        Returns
        -------
        batches : Rollout
            Sampled mixed batches, each leaf shaped
            `(P, n_updates, batch_size, T, *)`.
        """
        n_fresh_per_update = max(1, round(batch_size * (1.0 - self.replay_ratio)))
        n_fresh_per_meta = n_updates * n_fresh_per_update
        n_replay_per_update = batch_size - n_fresh_per_update
        num_trainers = self.storage.actions.shape[0]

        return _sample_compiled(
            self.storage,
            self.write_idx,
            self.valid_count,
            rng,
            num_trainers,
            n_updates,
            n_fresh_per_update,
            n_replay_per_update,
            n_fresh_per_meta,
            self.capacity,
        )

    def reset_at(self, idx: int) -> Self:
        """
        Clear the replay state for a single agent.

        Zeros that agent's storage slice and resets its `valid_count`
        to `0`. The shared `write_idx` is left untouched — newly
        collected trajectories will populate the reset agent's slots
        in lockstep with all the others.

        Parameters
        ----------
        idx : int
            Trainer index to clear.

        Returns
        -------
        new_buffer : MixedBuffer
            Updated buffer with agent `idx` cleared.
        """
        new_storage = jax.tree.map(
            lambda x: x.at[idx].set(jnp.zeros_like(x[idx])),
            self.storage,
        )
        new_valid_count = self.valid_count.at[idx].set(0)
        return self.__replace__(
            storage=new_storage,
            valid_count=new_valid_count,
        )

    def memory_mb(self) -> float:
        """
        Total size of all pre-allocated buffers in MB.

        Returns
        -------
        size_mb : float
            Combined size of all JAX buffers in megabytes.
        """
        return sum(a.nbytes for a in jax.tree.leaves(self.storage)) / 1e6

    def __repr__(self) -> str:
        P, capacity, T = jnp.shape(self.storage.actions)[:3]
        A = jnp.shape(self.storage.actions)[3]
        return (
            f"MixedBuffer("
            f"trainers={P}, capacity={capacity}, seq_len={T}, "
            f"max_action_dim={A}, replay_ratio={self.replay_ratio}, "
            f"memory={self.memory_mb():.1f}MB"
            f")"
        )
