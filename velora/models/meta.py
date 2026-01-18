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

from velora.config.outputs import DiscoPredictions
from velora.config.spec import DiscoHeadSpec, NCPWiringSpec
from velora.models.lnn.base import BaseNCP
from velora.models.lnn.wiring import NCPWiringBuilder


class DiscoNetwork(BaseNCP):
    """
    Meta-network that produces learned targets (π̂, ŷ, ẑ) for the update rule. Uses practices from the DiscoRL paper.

    Uses a Liquid Neural Network (LNN) architecture with 3 output heads and a linear action gate:

        1. Policy targets: `π̂,(s, a)`
        2. Observation-conditioned targets: `ŷ(s)`
        3. Action-conditioned targets: `ẑ(s, a)`
        4. Action gate: modulates `π̂,` with per-action embeddings

    Parameters
    ----------
    n_inputs : int
        Number of input nodes (flattened agent output + env signals)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    n_actions : int
        Size of the policy targets `π̂,`
    prediction_size : int
        Size of the target vectors `(ŷ, ẑ)`
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
        n_actions: int,
        prediction_size: int,
        action_embed_dim: int,
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
    ) -> None:
        self.n_actions = n_actions
        self.pred_size = prediction_size
        self.action_embed_dim = action_embed_dim

        super().__init__(
            n_inputs,
            n_neurons,
            key=key,
            sparsity=sparsity,
        )

        self.action_gate = nnx.Linear(
            self.action_embed_dim,
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
                pi=self.n_actions,
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
            - `n_actions (A)`: the number of discrete actions
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
        target_preds : TargetPredictions
            Network predictions for the command layer (`embedding`) and each head `(π̂, ŷ, ẑ)`
        h_state : chex.Array
            Final hidden state with shape `(B, H)`
        """
        x, h_state, timespans = self._preprocess(obs, h_state, timespans)
        h_state, preds = self._scan(x, h_state, timespans, reverse=True)

        # 4 outputs -> (embedding, π̂, ŷ, ẑ)
        embedding, pi_raw, y, z = self._postprocess(preds)

        # Modulate pi with action embeddings
        pi = self._modulate_pi(pi_raw, action_emb)

        return DiscoPredictions(embedding=embedding, pi=pi, y=y, z=z), h_state

    def _modulate_pi(
        self,
        pi_raw: chex.Array,
        action_emb: chex.Array,
    ) -> chex.Array:
        """
        Modulate policy targets with per-action embeddings.

        Applies a learned gate derived from action embeddings to adjust the raw policy targets on a per-action basis.

        Parameters
        ----------
        pi_raw : chex.Array
            Raw policy targets from pi_head. Shape: `(B, T, A)`
        action_emb : chex.Array
            Per-action embeddings. Shape: `(B, T, A, C)`

        Returns
        -------
        pi : chex.Array
            Modulated policy targets. Shape: `(B, T, A)`
        """
        # Compute per-action gate
        # (B, T, A, C) -> (B, T, A, 1) -> (B, T, A)
        gate = self.action_gate(action_emb)  # type: ignore
        gate = jnp.squeeze(gate, axis=-1)
        gate = nnx.sigmoid(gate)

        # Modulate raw targets
        pi = pi_raw * gate
        return pi
