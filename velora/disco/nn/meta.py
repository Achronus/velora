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
from flax import nnx

from velora.disco.outputs import DiscoPredictions
from velora.disco.spec import DiscoHeadSpec
from velora.lnn.base import BaseCfC
from velora.lnn.spec import NCPWiringSpec
from velora.lnn.wiring import NCPWiringBuilder


class DiscoNetwork(BaseCfC):
    """
    Meta-network that produces learned targets `(π̂, ŷ, ẑ)` for the update rule. Uses practices from the DiscoRL paper.

    Uses a Liquid Neural Network (LNN) architecture with action-agnostic outputs:

        1. Policy targets: `π̂,(s, a)`
        2. Observation-conditioned targets: `ŷ(s)`
        3. Action-conditioned targets: `ẑ(s, a)`

    The network derives `n_actions` from input shapes at runtime, allowing the same
    trained weights to work across environments with different action spaces.

    Parameters
    ----------
    n_inputs : int
        Number of input nodes (flattened agent output + env signals)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    prediction_size : int
        Size of the target vectors `(ŷ, ẑ)` and policy hidden representation
    action_embed_dim : int
        Dimension of per-action embeddings from encoder
    key : chex.PRNGKey
        Random number generator key
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Default is `0.5`.

        Must be a value between `[0.1, 0.9]`:

            - Where `0.1` neurons are very dense
            - Where `0.9` neurons are very sparse
    """

    motor: DiscoHeadSpec  # type: ignore

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        prediction_size: int,
        action_embed_dim: int,
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
    ) -> None:
        self.pred_size = prediction_size
        self.action_embed_dim = action_embed_dim

        super().__init__(
            n_inputs,
            n_neurons,
            key=key,
            sparsity=sparsity,
        )

        self.policy_proj = nnx.Linear(
            self.pred_size + self.action_embed_dim,
            1,
            rngs=self.rngs,
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
                DiscoHeadSpec,
                pi=self.pred_size,
                y=self.pred_size,
                z=self.pred_size,
            )
            .build()
        )

    def __call__(
        self,
        obs: chex.Array,
        action_emb: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[DiscoPredictions, chex.Array]:
        """
        Forward pass through the network.

        Parameters
        ----------
        obs : chex.Array
            Input observations with shape `(B, T, F)` or `(B, F)`

            - `batch_size (B)`: the number of samples per timestep
            - `seq_length (T)`: the number of sequences (e.g., trajectories)
            - `features (F)`: the features at each timestep
        action_emb : chex.Array
            Per-action embeddings from encoder with shape `(B, T, A, C)`

            - `batch_size (B)`: the number of samples per timestep
            - `seq_length (T)`: the number of sequences (e.g., trajectories)
            - `n_actions (A)`: the number of discrete actions (derived at runtime)
            - `action_embed_dim (C)`: the action embedding dimension
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
        target_preds : DiscoPredictions
            Network predictions for the command layer (`embedding`) and each head `(π̂, ŷ, ẑ)`
        h_state : chex.Array
            Final hidden state with shape `(B, H)`
        """
        x, h_state, timespans = self._preprocess(obs, h_state, timespans)
        h_state, preds = self._scan(x, h_state, timespans, reverse=True)

        # 4 outputs -> (embedding, pi_hidden, ŷ, ẑ)
        embedding, pi_hidden, y, z = self._postprocess(preds)

        # Generate per-action policy targets using action embeddings
        # n_actions is derived from action_emb shape
        pi = self._compute_policy_targets(pi_hidden, action_emb)

        return DiscoPredictions(embedding=embedding, pi=pi, y=y, z=z), h_state

    def _compute_policy_targets(
        self,
        pi_hidden: chex.Array,
        action_emb: chex.Array,
    ) -> chex.Array:
        """
        Compute per-action policy targets from hidden representation and action embeddings.

        Uses a conv1d-style approach where the same learned weights are applied
        independently to each action, making the output dimension dynamic based
        on the number of actions in the environment.

        Parameters
        ----------
        pi_hidden : chex.Array
            Policy hidden representation from network. Shape: `(B, T, H)`
        action_emb : chex.Array
            Per-action embeddings from encoder. Shape: `(B, T, A, C)`

        Returns
        -------
        pi : chex.Array
            Per-action policy targets. Shape: `(B, T, A)`
        """
        # Get n_actions from action_emb shape (dynamic)
        n_actions = jnp.shape(action_emb)[2]

        # Broadcast pi_hidden to match action dimension
        # (B, T, H) -> (B, T, 1, H) -> (B, T, A, H)
        pi_expanded = jnp.expand_dims(pi_hidden, axis=2)
        pi_broadcast = jnp.repeat(pi_expanded, n_actions, axis=2)

        # Concatenate with action embeddings: (B, T, A, H + C)
        combined = jnp.concatenate([pi_broadcast, action_emb], axis=-1)

        # Apply shared linear projection per-action (conv1d with kernel=1)
        # (B, T, A, H + C) -> (B, T, A, 1) -> (B, T, A)
        pi = self.policy_proj(combined)  # type: ignore
        pi = jnp.squeeze(pi, axis=-1)

        return pi
