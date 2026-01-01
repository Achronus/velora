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

from typing import Optional, Tuple

import chex
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from velora.models.lnn.cell import NCPLiquidCell
from velora.models.lnn.wiring import (
    ACMHeadConfig,
    OCMHeadConfig,
    build_acm_wiring,
    build_ocm_wiring,
)
from velora.utils.nn import active_parameters, total_parameters


class ACM(nnx.Module):
    """
    An Action-Conditional Model (ACM) used to enable action-aware learning.

    Uses a Liquid Neural Network (LNN) architecture with 3 output heads:

        1. Action-conditioned prediction: z(s, a) - Action-conditioned
        prediction for control-relevant targets.
        2. Auxiliary policy prediction: p(s, a) - Auxiliary policy prediction
        for representation learning.
        3. Action-value: q(s, a) - Action-value for value-based bootstrapping.

    Parameters:
        obs_dim (int): number of observations (sensory nodes)
        n_neurons (int): number of decision nodes (inter + command nodes)
        prediction_size (int): size of the action-conditioned prediction vector
        n_actions (int): number of discrete actions
        q_dim (int): dimension of action-value prediction head. Uses distributional Q-values
        key (jax.random.PRNGKey): random number generator key
        sparsity_level (float, optional): network connection sparsity
            between neurons. Default is `0.5`.
    """

    def __init__(
        self,
        obs_dim: int,
        n_neurons: int,
        prediction_size: int,
        n_actions: int,
        q_dim: int,
        *,
        key: chex.PRNGKey,
        sparsity_level: float = 0.5,
    ) -> None:
        self.z_dim = prediction_size
        self.n_actions = n_actions  # aux_pi
        self.q_dim = q_dim

        self.obs_dim = obs_dim + self.n_actions
        self.n_neurons = n_neurons
        self.key = key

        self.seed = jax.random.key_data(key)[-1].item()
        self.rngs = nnx.Rngs(params=self.key)

        self.wiring = nnx.data(
            build_acm_wiring(
                self.obs_dim,
                self.n_neurons,
                self.z_dim,
                self.n_actions,
                self.q_dim,
                seed=self.seed,
                sparsity_level=sparsity_level,
            )
        )

        self.motor: ACMHeadConfig = nnx.data(self.wiring.motor)  # type: ignore

        self.hidden_size = (
            self.wiring.inter.n_hidden
            + self.wiring.command.n_hidden
            + self.wiring.motor.hidden_count()
        )

        # Inter layer: sensory -> inter
        self.inter = NCPLiquidCell(
            self.obs_dim,
            self.wiring.inter.n_hidden,
            self.wiring.inter.mask,
            rngs=self.rngs,
        )

        # Command layer: inter -> command
        self.command = NCPLiquidCell(
            self.wiring.inter.n_hidden,
            self.wiring.command.n_hidden,
            self.wiring.command.mask,
            rngs=self.rngs,
        )

        # Motor layers: command -> motors (outputs)
        self.z_head = NCPLiquidCell(
            self.wiring.command.n_hidden,
            self.motor.z.n_hidden,
            self.motor.z.mask,
            rngs=self.rngs,
        )
        self.aux_pi_head = NCPLiquidCell(
            self.wiring.command.n_hidden,
            self.motor.aux_pi.n_hidden,
            self.motor.aux_pi.mask,
            rngs=self.rngs,
        )
        self.q_head = NCPLiquidCell(
            self.wiring.command.n_hidden,
            self.motor.q.n_hidden,
            self.motor.q.mask,
            rngs=self.rngs,
        )

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns:
            count (int): the total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns:
            count (int): the active parameter count.
        """
        return self._active_params

    def _split_h_state(
        self, h: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Helper method. Splits the NCPs hidden state into layer-specific states.

        Parameters:
            h (jax.Array): the network hidden state

        Returns:
            h_split (Tuple[chex.Array, ...]): hidden state split
            into layers `(inter, command, z, aux, q)`
        """
        split_indices = jnp.cumsum(
            jnp.array(
                [
                    self.wiring.inter.n_hidden,
                    self.wiring.command.n_hidden,
                    self.motor.z.n_hidden,
                    self.motor.aux_pi.n_hidden,
                ]
            )
        )
        h_inter, h_command, h_z, h_aux, h_q = jnp.split(h, split_indices, axis=1)
        return h_inter, h_command, h_z, h_aux, h_q

    def _encode_obs_with_actions(self, state: chex.Array) -> chex.Array:
        """
        Expands the input observation with one-hot encoded actions (A)
        using an identity matrix for all batches.

        Parameters:
            state (jax.Array): state embedding with shape `(B, F, T)`

        Returns:
            state_with_actions (jax.Array): obs with batched one-hot encoded
            actions in the shape `(B*A, F+A, T)`
        """
        B, F, T = jnp.shape(state)

        # Expand state for all actions: (B, F, T) -> (BA, F, T)
        state_expanded = jnp.repeat(state, self.n_actions, axis=0)

        # Create one-hot actions: (BA, A) -> (BA, A, T)
        one_hot_actions = jnp.eye(self.n_actions)  # (A, A)
        one_hot_actions = jnp.tile(one_hot_actions, [B, 1])  # (BA, A)
        one_hot_actions = jnp.expand_dims(one_hot_actions, -1)  # (BA, A, 1)
        one_hot_actions = jnp.tile(one_hot_actions, [1, 1, T])  # (BA, A, T)

        return jnp.concatenate(
            [state_expanded, one_hot_actions],
            axis=1,
        )  # (BA, F+A, T)

    def __call__(
        self,
        state_embedding: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Performs a forward pass through the network.

        Parameters:
            state_embedding (jax.Array): embedded state from Encoder
                `(B, F, T)` or `(B, F)`.

                - `batch_size (B)` the number of samples per timestep.
                - `features (F)` the features at each timestep
                - `seq_length (T)` the number of sequences (e.g., trajectories).
            h_state (jax.Array, optional): initial hidden state `(B, H)`.

                - `batch_size (B)` the number of samples per timestep.
                - `n_hidden (H)` the total number of hidden neurons.

            timespans (jax.Array, optional): time elapsed since previous
                timestep. For fixed intervals set to `None`.
                For varying timesteps shape must be `(T,)`

                - `seq_length (T)` the number of sequences (e.g., trajectories).
        Returns:
            z (jax.Array): action-conditioned prediction `(B, A, F, T)`.
            aux_pi (jax.Array): auxiliary policy prediction `(B, A, F, T)`.
            q (jax.Array): action-value prediction `(B, A, F, T)`.
            h_state (jax.Array): final hidden state `(B*A, H)`.
        """
        if state_embedding.ndim == 2:
            state_embedding = jnp.expand_dims(state_embedding, -1)  # Add time dim

        B, F, T = jnp.shape(state_embedding)

        # Expand with action encodings
        x = self._encode_obs_with_actions(state_embedding)  # (BA, F+A, T)

        if h_state is None:
            h_state = jnp.zeros((B * self.n_actions, self.hidden_size))  # (BA, H)

        timespans = jnp.ones(T) if timespans is None else timespans

        def _step(
            h: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
            inputs: Tuple[chex.Array, chex.Array],
        ) -> Tuple[
            Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array, chex.Array],
        ]:
            """Single step function."""
            h_inter, h_command, h_z, h_aux, h_q = h
            x_t, ts_t = inputs  # x_t -> (BA, F+A), ts_t -> scalar

            # Forward through each liquid layer
            x_t, new_h_inter = self.inter(x_t, h_inter, ts_t)
            x_t, new_h_command = self.command(x_t, h_command, ts_t)

            z_t, new_h_z = self.z_head(x_t, h_z, ts_t)  # z_t -> (B, F)
            aux_t, new_h_aux = self.aux_pi_head(x_t, h_aux, ts_t)  # aux_t -> (B, F)
            q_t, new_h_q = self.q_head(x_t, h_q, ts_t)  # q_t -> (B, F)

            new_h = (new_h_inter, new_h_command, new_h_z, new_h_aux, new_h_q)
            preds = (z_t, aux_t, q_t)
            return new_h, preds

        # Transpose for scanning over time: (BA, F+A, T) -> (T, BA, F+A)
        x_transposed = jnp.transpose(x, (2, 0, 1))

        # Split hidden states for each layer
        h_split = self._split_h_state(h_state)
        scan_inputs = (x_transposed, timespans)

        new_h, preds = jax.lax.scan(_step, h_split, scan_inputs, length=T)

        h_state = jnp.concatenate(new_h, axis=1)  # (BA, H)
        z, aux_pi, q = preds

        # Transpose back: (T, BA, F) -> (BA, F, T)
        z = jnp.transpose(z, (1, 2, 0))
        aux_pi = jnp.transpose(aux_pi, (1, 2, 0))
        q = jnp.transpose(q, (1, 2, 0))

        # Reshape to separate batch and action dims: (BA, F, T) -> (B, A, F, T)
        z = z.reshape(B, self.n_actions, -1, T)
        aux_pi = aux_pi.reshape(B, self.n_actions, -1, T)
        q = q.reshape(B, self.n_actions, -1, T)

        return z, aux_pi, q, h_state


class OCM(nnx.Module):
    """
    An Observation-Conditional Model (OCM) used to encode observations and
    capture state-level information that is usable by the meta-network.

    Uses a Liquid Neural Network (LNN) architecture with 2 output heads:

        1. Policy: π(s, a) - policy logits for action probabilities.
        2. Observation-conditioned prediction: y(s) - state-level
        features with discovered semantics.

    Parameters:
        obs_dim (int): number of observations (sensory nodes)
        n_neurons (int): number of decision nodes (inter + command nodes)
        prediction_size (int): size of the observation-conditioned prediction vector
        n_actions (int): number of discrete actions
        key (jax.random.PRNGKey): random number generator key
        sparsity_level (float, optional): network connection sparsity
            between neurons. Default is `0.5`.
    """

    def __init__(
        self,
        obs_dim: int,
        n_neurons: int,
        prediction_size: int,
        n_actions: int,
        *,
        key: chex.PRNGKey,
        sparsity_level: float = 0.5,
    ) -> None:
        self.y_dim = prediction_size
        self.n_actions = n_actions  # pi

        self.obs_dim = obs_dim
        self.n_neurons = n_neurons
        self.key = key

        self.seed = jax.random.key_data(key)[-1].item()
        self.rngs = nnx.Rngs(params=self.key)

        self.wiring = nnx.data(
            build_ocm_wiring(
                self.obs_dim,
                n_neurons,
                self.y_dim,
                self.n_actions,
                seed=self.seed,
                sparsity_level=sparsity_level,
            )
        )

        self.motor: OCMHeadConfig = nnx.data(self.wiring.motor)  # type: ignore

        self.hidden_size = (
            self.wiring.inter.n_hidden
            + self.wiring.command.n_hidden
            + self.motor.hidden_count()
        )
        self.embedding_size = self.wiring.command.n_hidden

        # Inter layer: sensory -> inter
        self.inter = NCPLiquidCell(
            self.obs_dim,
            self.wiring.inter.n_hidden,
            self.wiring.inter.mask,
            rngs=self.rngs,
        )

        # Command layer: inter -> command
        self.command = NCPLiquidCell(
            self.wiring.inter.n_hidden,
            self.wiring.command.n_hidden,
            self.wiring.command.mask,
            rngs=self.rngs,
        )

        # Motor layers: command -> motors (outputs)
        self.pi_head = NCPLiquidCell(
            self.wiring.command.n_hidden,
            self.motor.pi.n_hidden,
            self.motor.pi.mask,
            rngs=self.rngs,
        )
        self.y_head = NCPLiquidCell(
            self.wiring.command.n_hidden,
            self.motor.y.n_hidden,
            self.motor.y.mask,
            rngs=self.rngs,
        )

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns:
            count (int): the total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns:
            count (int): the active parameter count.
        """
        return self._active_params

    def _split_h_state(
        self, h: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Helper method. Splits the NCPs hidden state into layer-specific states.

        Parameters:
            h (jax.Array): the network hidden state

        Returns:
            h_split (Tuple[chex.Array, ...]): hidden state split
            into layers `(inter, command, pi, y, )`
        """
        split_indices = jnp.cumsum(
            jnp.array(
                [
                    self.wiring.inter.n_hidden,
                    self.wiring.command.n_hidden,
                    self.motor.pi.n_hidden,
                ]
            )
        )
        h_inter, h_command, h_pi, h_y = jnp.split(h, split_indices, axis=1)
        return h_inter, h_command, h_pi, h_y

    def __call__(
        self,
        obs: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Forward pass through the network.

        Parameters:
            obs (jax.Array): input observations `(B, F, T)` or `(B, F)`.

                - `batch_size (B)` the number of samples per timestep.
                - `features (F)` the features at each timestep
                - `seq_length (T)` the number of sequences (e.g., trajectories).
            h_state (jax.Array, optional): initial hidden state `(B, H)`.

                - `batch_size (B)` the number of samples per timestep.
                - `n_hidden (H)` the total number of hidden neurons.

            timespans (jax.Array, optional): time elapsed since previous
                timestep. For fixed intervals set to `None`.
                For varying timesteps shape must be `(T,)`

                - `seq_length (T)` the number of sequences (e.g., trajectories).
        Returns:
            pi (jax.Array): policy prediction `(B, F, T)`.
            y (jax.Array): observation-conditioned prediction `(B, F, T)`.
            embedding (jax.Array): command layer output `(B, F, T)`.
                Provided to ACM as input.
            h_state (jax.Array): final hidden state `(B, H)`.
        """
        if obs.ndim == 2:
            obs = jnp.expand_dims(obs, -1)  # Add time dim

        B, F, T = jnp.shape(obs)

        if h_state is None:
            h_state = jnp.zeros((B, self.hidden_size))

        timespans = jnp.ones(T) if timespans is None else timespans

        def _step(
            h: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
            inputs: Tuple[chex.Array, chex.Array],
        ) -> Tuple[
            Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array, chex.Array],
        ]:
            """Single step function."""
            h_inter, h_command, h_pi, h_y = h
            x_t, ts_t = inputs  # x_t -> (B, F), ts_t -> scalar

            # Forward through each liquid layer
            x_t, new_h_inter = self.inter(x_t, h_inter, ts_t)
            embed_t, new_h_command = self.command(x_t, h_command, ts_t)

            pi_t, new_h_pi = self.pi_head(embed_t, h_pi, ts_t)  # pi_t -> (B, F)
            y_t, new_h_y = self.y_head(embed_t, h_y, ts_t)  # y_t -> (B, F)

            new_h = (new_h_inter, new_h_command, new_h_pi, new_h_y)
            preds = (pi_t, y_t, embed_t)
            return new_h, preds

        # Transpose for scanning over time: (B, F, T) -> (T, B, F)
        x_transposed = jnp.transpose(obs, (2, 0, 1))

        # Split hidden states for each layer
        h_split = self._split_h_state(h_state)
        scan_inputs = (x_transposed, timespans)

        new_h, preds = jax.lax.scan(_step, h_split, scan_inputs, length=T)

        h_state = jnp.concatenate(new_h, axis=1)  # (B, H)
        pi, y, embedding = preds

        # Transpose back: (T, B, F) -> (B, F, T)
        pi = jnp.transpose(pi, (1, 2, 0))
        y = jnp.transpose(y, (1, 2, 0))
        embedding = jnp.transpose(embedding, (1, 2, 0))

        return pi, y, embedding, h_state
