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

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from velora.config.outputs import BufferSamples, PolicyAgentOutput
from velora.config.settings import MixedBufferSettings


class MixedBuffer:
    """
    A circular buffer for storing and sampling trajectories.

    Supports mixed sampling of replay (off-policy) and rollout (on-policy) data.

    Lazily initializes agent output arrays on first `add()` call, inferring
    shapes directly from the provided `AgentOutput`.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key
    config : MixedBufferSettings (optional)
        Configuration for the buffer. Default is `MixedBufferSettings()`
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        *,
        config: MixedBufferSettings = MixedBufferSettings(),
    ) -> None:
        self.key = key
        self.seq_len = config.seq_len
        self.capacity = config.capacity
        self.split_ratio = config.split_ratio

        # Core data
        self.actions = jnp.zeros((self.capacity, self.seq_len, 1), dtype=jnp.int32)
        self.rewards = jnp.zeros((self.capacity, self.seq_len, 1))
        self.discounts = jnp.zeros((self.capacity, self.seq_len, 1))
        self.values = jnp.zeros((self.capacity, self.seq_len, 1))

        # Agent outputs (placeholders)
        self.pi = jnp.zeros(1)
        self.y = jnp.zeros(1)
        self.z = jnp.zeros(1)
        self.aux_pi = jnp.zeros(1)
        self.q = jnp.zeros(1)

        self.target_pi = jnp.zeros(1)
        self.target_y = jnp.zeros(1)
        self.target_z = jnp.zeros(1)
        self.target_aux_pi = jnp.zeros(1)
        self.target_q = jnp.zeros(1)

        # Buffer status
        self.ptr = 0
        self.size = 0

    @property
    def is_initialized(self) -> bool:
        """
        Checks if buffer arrays are initialized.

        Returns
        -------
        status : bool
            True if initialized. Otherwise, False
        """
        return jnp.shape(self.pi) != (1,)

    def _set_array(self, dims: Tuple[int, ...]) -> jax.Array:
        """
        Helper method that initializes a buffer array given a set of dimensions.

        Creates an array with shape: `(N, T, *dims)`.

        Parameters
        ----------
        dims : Tuple[int, ...]
            Trailing dimensions for the array

        Returns
        -------
        array : jax.Array
            A zero-initialized array
        """
        N = self.capacity
        T = self.seq_len

        return jnp.zeros((N, T, *dims))

    def _lazy_init(self, preds: PolicyAgentOutput) -> None:
        """
        Helper method that initializes agent output arrays.

        Infers shapes from first sample.

        Parameters
        ----------
        preds : AgentOutput
            First agent predictions from the policy network
        """

        def _trailing_dims(x: chex.Array) -> Tuple[int, ...]:
            """Helper method. Extracts dimensions after B and T."""
            # (B, T, ...) -> (...)
            return jnp.shape(x)[2:]

        # Extract trailing dimensions from each array
        pi_dims = _trailing_dims(preds.pi)  # (B, T, A) -> (A,)
        y_dims = _trailing_dims(preds.y)  # (B, T, Y) -> (Y,)
        z_dims = _trailing_dims(preds.z)  # (B, T, A, Z) -> (A, Z)
        aux_pi_dims = _trailing_dims(preds.aux_pi)  # (B, T, A, A) -> (A, A)
        q_dims = _trailing_dims(preds.q)  # (B, T, A, Q) -> (A, Q)

        # Init arrays: (N, T, *dims)
        self.pi = self._set_array(pi_dims)  # (N, T, A)
        self.y = self._set_array(y_dims)  # (N, T, Y)
        self.z = self._set_array(z_dims)  # (N, T, A, Z)
        self.aux_pi = self._set_array(aux_pi_dims)  # (N, T, A, A)
        self.q = self._set_array(q_dims)  # (N, T, A, Q)

        self.target_pi = self._set_array(pi_dims)
        self.target_y = self._set_array(y_dims)
        self.target_z = self._set_array(z_dims)
        self.target_aux_pi = self._set_array(aux_pi_dims)
        self.target_q = self._set_array(q_dims)

    def add(
        self,
        actions: chex.Array,
        rewards: chex.Array,
        discounts: chex.Array,
        values: chex.Array,
        preds: PolicyAgentOutput,
        target_preds: PolicyAgentOutput,
    ) -> None:
        """
        Adds a batch of trajectories to the buffer.

        On the first call, initializes agent output arrays by inferring shapes
        from `preds`.

        Parameters
        ----------
        actions : jax.Array
            Actions taken in the environment `(B, T)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of timesteps in the trajectory

        rewards : jax.Array
            Rewards generated from the environment `(B, T)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of timesteps in the trajectory

        discounts : jax.Array
            Environment discounts `(B, T)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of timesteps in the trajectory
            - Binary values: `1.0` = episode continues, `0.0` = episode ended

        values : jax.Array
            State value estimates `(B, T)`.

            - batch_size (`B`) - the number of samples per timestep.
            - seq_length (`T`) - the number of timesteps in the trajectory.

        preds : AgentOutput
            Agent network outputs for the trajectory
        target_preds : AgentOutput
            Agent target network outputs for the trajectory
        """
        if not self.is_initialized:
            self._lazy_init(preds)

        batch_size = jnp.shape(actions)[0]
        indices = (self.ptr + np.arange(batch_size)) % self.capacity

        # Update core data
        self.actions = self.actions.at[indices].set(actions)
        self.rewards = self.rewards.at[indices].set(rewards)
        self.discounts = self.discounts.at[indices].set(discounts)
        self.values = self.values.at[indices].set(values)

        # Update agent outputs
        self.pi = self.pi.at[indices].set(preds.pi)
        self.y = self.y.at[indices].set(preds.y)
        self.z = self.z.at[indices].set(preds.z)
        self.aux_pi = self.aux_pi.at[indices].set(preds.aux_pi)
        self.q = self.q.at[indices].set(preds.q)

        self.target_pi = self.target_pi.at[indices].set(target_preds.pi)
        self.target_y = self.target_y.at[indices].set(target_preds.y)
        self.target_z = self.target_z.at[indices].set(target_preds.z)
        self.target_aux_pi = self.target_aux_pi.at[indices].set(target_preds.aux_pi)
        self.target_q = self.target_q.at[indices].set(target_preds.q)

        # Update buffer status
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> BufferSamples:
        """
        Sample a mixed batch of replay and rollout trajectories.

        Parameters
        ----------
        batch_size : int
            Total number of trajectories to sample

        Returns
        -------
        samples : BufferSamples
            Sampled trajectories
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Buffer not initialized. Call `add()` at least once before sampling."
            )

        replay_size = int(batch_size * self.split_ratio)
        rollout_size = batch_size - replay_size

        end_idx = self.ptr - 1

        key1, key2, key3 = jax.random.split(self.key, 3)

        def _get_indices(key: chex.Array, size1: int, size2: int) -> chex.Array:
            """Helper method. Gets buffer indices."""
            return jax.random.randint(key, (size1,), 0, size2)

        # Randomly sample from buffer
        replay_indices = _get_indices(key1, replay_size, self.size)
        rollout_indices = (
            end_idx - _get_indices(key2, rollout_size, rollout_size)
        ) % self.capacity

        # Combine and shuffle
        indices = jnp.concatenate([replay_indices, rollout_indices])
        indices = jax.random.permutation(key3, indices)

        return BufferSamples(
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            discounts=self.discounts[indices],
            values=self.values[indices],
            preds=PolicyAgentOutput(
                pi=self.pi[indices],
                y=self.y[indices],
                z=self.z[indices],
                aux_pi=self.aux_pi[indices],
                q=self.q[indices],
            ),
            target_preds=PolicyAgentOutput(
                pi=self.target_pi[indices],
                y=self.target_y[indices],
                z=self.target_z[indices],
                aux_pi=self.target_aux_pi[indices],
                q=self.target_q[indices],
            ),
        )

    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough trajectories to sample.

        Parameters
        ----------
        min_size : int
            Minimum required trajectories

        Returns
        -------
        status : bool
            True if buffer has at least `min_size` trajectories
        """
        return self.is_initialized and self.size >= min_size

    def clear(self) -> None:
        """
        Clear the buffer by resetting the pointer and size.

        Note: This keeps allocated memory.
        """
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size
