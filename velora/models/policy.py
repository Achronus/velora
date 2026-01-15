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
import jax
import jax.numpy as jnp

from velora.config.outputs import ACMPredictions, OCMPredictions
from velora.config.spec import ACMHeadSpec, NCPWiringSpec, OCMHeadSpec
from velora.models.lnn.base import BaseNCP
from velora.models.lnn.wiring import NCPWiringBuilder


class ACM(BaseNCP):
    """
    An Action-Conditional Model (ACM) used to enable action-aware learning.

    Uses a Liquid Neural Network (LNN) architecture with 3 output heads:

        1. Action-conditioned prediction: z(s, a) - Action-conditioned
            prediction for control-relevant targets
        2. Auxiliary policy prediction: p(s, a) - Auxiliary policy prediction
           for representation learning
        3. Action-value: q(s, a) - Action-value for value-based bootstrapping

    Parameters
    ----------
    obs_dim : int
        Number of observations (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    prediction_size : int
        Size of the action-conditioned prediction vector
    n_actions : int
        Number of discrete actions
    q_dim : int
        Dimension of action-value prediction head. Uses distributional Q-values
    key : chex.PRNGKey
        Random number generator key
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Default is `0.5`.

        Must be a value between `[0.1, 0.9]`:

            - Where `0.1` neurons are very dense
            - Where `0.9` neurons are very sparse
    """

    motor: ACMHeadSpec  # type: ignore

    def __init__(
        self,
        obs_dim: int,
        n_neurons: int,
        prediction_size: int,
        n_actions: int,
        q_dim: int,
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
    ) -> None:
        self.z_dim = prediction_size
        self.n_actions = n_actions  # aux_pi
        self.q_dim = q_dim

        super().__init__(
            obs_dim + self.n_actions,
            n_neurons,
            key=key,
            sparsity=sparsity,
        )

    def _build_wiring(self) -> NCPWiringSpec:
        return (
            NCPWiringBuilder(
                self.in_features,
                self.n_neurons,
                seed=self.seed,
                sparsity=self.sparsity,
            )
            .add_output_heads(
                ACMHeadSpec,
                z=self.z_dim,
                aux_pi=self.n_actions,
                q=self.q_dim,
            )
            .build()
        )

    def _encode_obs_with_actions(self, state: chex.Array) -> chex.Array:
        """
        Helper method. Expands the input observation with one-hot
        encoded actions (A) using an identity matrix for all batches.

        Parameters
        ----------
        state : chex.Array
            State embedding with shape `(B, T, F)`

        Returns
        -------
        state_with_actions : chex.Array
            Obs with batched one-hot encoded actions in the shape `(B*A, T, F+A)`
        """
        B, T, F = jnp.shape(state)

        # Expand state for all actions: (B, T, F) -> (BA, T, F)
        state_expanded = jnp.repeat(state, self.n_actions, axis=0)

        # Create one-hot actions: (A,) -> (BA, T, A)
        one_hot_actions = jnp.eye(self.n_actions)  # (A, A)
        one_hot_actions = jnp.tile(one_hot_actions, [B, 1])  # (BA, A)
        one_hot_actions = jnp.expand_dims(one_hot_actions, axis=1)  # (BA, 1, A)
        one_hot_actions = jnp.tile(one_hot_actions, [1, T, 1])  # (BA, T, A)

        return jnp.concatenate(
            [state_expanded, one_hot_actions],
            axis=-1,
        )  # (BA, T, F+A)

    def _split_action_dim(self, x: chex.Array) -> chex.Array:
        """
        Helper method. Reshapes action-expanded array to
        separate batch and action dimensions.

        Converts from flattened `(B*A, T, F)` format back to
        `(B, T, A, F)` format after scan operations.

        Parameters
        ----------
        x : chex.Array
            Input array with shape `(B*A, T, F)`

        Returns
        -------
        new_x : chex.Array
            A reshaped `x` with shape `(B, T, A, F)`
        """
        BA, T, F = jnp.shape(x)
        B = BA // self.n_actions

        # (B*A, T, F) -> (B, A, T, F) -> (B, T, A, F)
        x = x.reshape(B, self.n_actions, T, F)
        x = jnp.transpose(x, axes=(0, 2, 1, 3))
        return x

    def _preprocess(
        self,
        x: chex.Array,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Preprocesses `__call__` method inputs.

        Includes -
        - `x` dimension expansion from `(B, F)` -> `(B, T, F)` (if needed)
        - `x` is then one-hot action encoded to `(B*A, T, F+A)`
        - `h_state` initialized to `(B*A, H)` when set to `None`
        - `timespans` initialized to `(T,)` of `1s` when set to `None`

        Parameters
        ----------
        x : jax.Array
            An input array of shape: `(B, F)` or `(B, T, F)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories, channels)
            - `features (F)` the features at each timestep

        h_state : jax.Array (optional)
            Initial hidden state of the RNN with shape: `(B, H)`

            - `batch_size (B)` the number of samples per timestep
            - `n_hidden (H)` the total number of hidden neurons

        timespans : jax.Array (optional)
            Time elapsed since previous timestep.
            For fixed intervals set to `None`. For varying timesteps shape
            should be `(T,)`
        """
        x, _, timespans = super()._preprocess(x, h_state, timespans)

        B, T, F = jnp.shape(x)
        x = self._encode_obs_with_actions(x)

        if h_state is None:
            h_state = jnp.zeros((B * self.n_actions, self.hidden_size))

        return x, h_state, timespans

    def _postprocess(self, preds: Tuple[chex.Array, ...]) -> Tuple[chex.Array, ...]:
        """
        Postprocess network predictions by applying -
            1. Batch-first transformations to all predictions
            2. Reshapes batch-first transforms from `(B*A, T, F)` to `(B, T, A, F)`

        Should be used in the `__call__` method after `_scan`.

        Parameters
        ----------
        preds : Tuple[chex.Array, ...]
            Raw predictions from scan `(T, B, F)`

        Returns
        -------
        outputs : Tuple[chex.Array, ...]
            Transformed predictions `(B, T, F)`
        """
        batch_first = super()._postprocess(preds)
        return jax.tree.map(self._split_action_dim, batch_first)

    def __call__(
        self,
        state_embedding: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[ACMPredictions, chex.Array]:
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        state_embedding : chex.Array
            Embedded state from Encoder with shape `(B, T, F)` or `(B, F)`

            - `batch_size (B)`: the number of samples per timestep
            - `seq_length (T)`: the number of sequences (e.g., trajectories)
            - `features (F)`: the features at each timestep
        h_state : chex.Array (optional)
            Initial hidden state with shape `(B, H)`

            - `batch_size (B)`: the number of samples per timestep
            - `n_hidden (H)`: the total number of hidden neurons
        timespans : chex.Array (optional)
            Time elapsed since previous timestep. For fixed intervals set to `None`.
            For varying timesteps shape must be `(T,)`

            - `seq_length (T)`: the number of sequences (e.g., trajectories)

        Returns
        -------
        ocm_preds : OCMPredictions
            Network predictions for the command layer (`embedding`) and each head `(z, aux_pi, q)`
        h_state : chex.Array
            Final hidden state with shape `(B*A, H)`
        """
        x, h_state, timespans = self._preprocess(state_embedding, h_state, timespans)
        h_state, preds = self._scan(x, h_state, timespans)

        # 4 outputs -> (embedding, z, aux_pi, q)
        embedding, z, aux_pi, q = self._postprocess(preds)
        return ACMPredictions(embedding=embedding, z=z, aux_pi=aux_pi, q=q), h_state


class OCM(BaseNCP):
    """
    An Observation-Conditional Model (OCM) used to encode observations and
    capture state-level information that is usable by the meta-network.

    Uses a Liquid Neural Network (LNN) architecture with 2 output heads:

        1. Policy: π(s, a) - policy logits for action probabilities.
        2. Observation-conditioned prediction: y(s) - state-level
           features with discovered semantics.

    Parameters
    ----------
    obs_dim : int
        Number of observations (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    prediction_size : int
        Size of the observation-conditioned prediction vector
    n_actions : int
        Number of discrete actions
    key : chex.PRNGKey
        Random number generator key
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Default is `0.5`.

        Must be a value between `[0.1, 0.9]`:

            - Where `0.1` neurons are very dense
            - Where `0.9` neurons are very sparse
    """

    motor: OCMHeadSpec  # type: ignore

    def __init__(
        self,
        obs_dim: int,
        n_neurons: int,
        prediction_size: int,
        n_actions: int,
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
    ) -> None:
        self.y_dim = prediction_size
        self.n_actions = n_actions  # pi

        super().__init__(
            obs_dim,
            n_neurons,
            key=key,
            sparsity=sparsity,
        )

        self.embedding_size = self.wiring.command.n_hidden

    def _build_wiring(self) -> NCPWiringSpec:
        return (
            NCPWiringBuilder(
                self.in_features,
                self.n_neurons,
                seed=self.seed,
                sparsity=self.sparsity,
            )
            .add_output_heads(
                OCMHeadSpec,
                pi=self.n_actions,
                y=self.y_dim,
            )
            .build()
        )

    def __call__(
        self,
        obs: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[OCMPredictions, chex.Array]:
        """
        Forward pass through the network.

        Parameters
        ----------
        obs : chex.Array
            Input observations with shape `(B, T, F)` or `(B, F)`

            - `batch_size (B)`: the number of samples per timestep
            - `seq_length (T)`: the number of sequences (e.g., trajectories)
            - `features (F)`: the features at each timestep
        h_state : chex.Array (optional)
            Initial hidden state with shape `(B, H)`

            - `batch_size (B)`: the number of samples per timestep
            - `n_hidden (H)`: the total number of hidden neurons
        timespans : chex.Array (optional)
            Time elapsed since previous timestep. For fixed intervals set to `None`.
            For varying timesteps shape must be `(T,)`

            - `seq_length (T)`: the number of sequences (e.g., trajectories)

        Returns
        -------
        ocm_preds : OCMPredictions
            Network predictions for the command layer (`embedding`) and each head `(pi, y)`
        h_state : chex.Array
            Final hidden state with shape `(B, H)`
        """
        x, h_state, timespans = self._preprocess(obs, h_state, timespans)
        h_state, preds = self._scan(x, h_state, timespans)

        # 3 outputs -> (embedding, pi, y)
        embedding, pi, y = self._postprocess(preds)
        return OCMPredictions(embedding=embedding, pi=pi, y=y), h_state
