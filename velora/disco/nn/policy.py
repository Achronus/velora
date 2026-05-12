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
import jax.numpy as jnp

from velora.disco.config.spec import ACMHeadSpec, OCMHeadSpec
from velora.disco.outputs import ACMPredictions, OCMPredictions
from velora.lnn.base import BaseCfC
from velora.lnn.spec import NCPWiringSpec
from velora.lnn.wiring import NCPWiringBuilder



class OCM(BaseCfC):
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
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
    ) -> None:
        self.y_dim = prediction_size
        self.pi_hidden_dim = prediction_size

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
                pi=self.pi_hidden_dim,
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
            Network predictions for the command layer (`embedding`)
            and each head `(pi, y)`
        h_state : chex.Array
            Final hidden state with shape `(B, H)`
        """
        x, h_state, timespans = self._preprocess(obs, h_state, timespans)
        h_state, preds = self._scan(x, h_state, timespans)

        # 3 outputs -> (embedding, pi, y)
        embedding, pi, y = self._postprocess(preds)
        return OCMPredictions(embedding=embedding, pi=pi, y=y), h_state


class ACM(BaseCfC):
    """
    Action-Conditional Model (ACM) for continuous action spaces.

    Takes the OCM state embedding concatenated with the continuous action
    vector as input, producing action-conditioned predictions `z(s, a)`,
    auxiliary policy predictions `aux_pi(s, a)`, and scalar Q-values `q(s, a)`.

    Uses a Liquid Neural Network (LNN) architecture with 3 output heads:

        1. Action-conditioned prediction: `z(s, a)` — learned representation
           with discovered semantics
        2. Auxiliary policy prediction: `aux_pi(s, a)` — predicts next-step
           policy parameters for representation learning
        3. Action-value: `q(s, a)` — scalar Q-value for the taken action

    Parameters
    ----------
    obs_dim : int
        Number of observations (OCM embedding size)
    max_action_dim : int
        Maximum continuous action dimensionality across all environments.
        Actions are zero-padded to this size for JIT-compatible static shapes.
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    prediction_size : int
        Size of the action-conditioned prediction vector `z` and
        auxiliary policy hidden representation
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
        max_action_dim: int,
        n_neurons: int,
        prediction_size: int,
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
    ) -> None:
        self.max_action_dim = max_action_dim
        self.z_dim = prediction_size
        self.aux_pi_dim = prediction_size
        self.q_dim = 1  # Scalar Q-value

        # Input: OCM embedding + continuous action vector
        super().__init__(
            obs_dim + max_action_dim,
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
                aux_pi=self.aux_pi_dim,
                q=self.q_dim,
            )
            .build()
        )

    def __call__(
        self,
        state_embedding: chex.Array,
        action: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[ACMPredictions, chex.Array]:
        """
        Performs a forward pass through the network.

        Concatenates the state embedding with the continuous action vector
        before processing through the LNN.

        Parameters
        ----------
        state_embedding : chex.Array
            Embedded state from OCM with shape `(B, T, F)` or `(B, F)`
        action : chex.Array
            Continuous action vector with shape `(B, T, D)` or `(B, D)`.
            Padded to `max_action_dim`.
        h_state : chex.Array (optional)
            Initial hidden state with shape `(B, H)`
        timespans : chex.Array (optional)
            Time elapsed since previous timestep `(T,)`

        Returns
        -------
        acm_preds : ACMPredictions
            Network predictions `(embedding, z, aux_pi, q)`
        h_state : chex.Array
            Final hidden state with shape `(B, H)`
        """
        # Concatenate state embedding with action
        x = jnp.concatenate([state_embedding, action], axis=-1)

        x, h_state, timespans = self._preprocess(x, h_state, timespans)
        h_state, preds = self._scan(x, h_state, timespans)

        # 4 outputs -> (embedding, z, aux_pi, q)
        embedding, z, aux_pi, q = self._postprocess(preds)
        return ACMPredictions(embedding=embedding, z=z, aux_pi=aux_pi, q=q), h_state
