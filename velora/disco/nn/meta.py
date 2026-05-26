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
from flax import nnx

from velora.disco.config.spec import DiscoHeadSpec
from velora.disco.outputs import DiscoPredictions
from velora.lnn.base import BaseCfC
from velora.lnn.spec import NCPWiringSpec
from velora.lnn.wiring import build_ncp_wiring


class DiscoNetwork(BaseCfC):
    """
    Meta-network that produces learned targets `(μ̂, log σ̂, ŷ, ẑ)` for the update rule. Suitable for continuous action spaces.

    Uses a Liquid Neural Network (LNN) architecture with action-agnostic outputs:

        1. Policy target predictions: `μ̂,(s, a)`, `log σ̂,(s, a)`
        2. Observation-conditioned targets: `ŷ(s)`
        3. Action-conditioned targets: `ẑ(s, a)`

    Parameters
    ----------
    n_inputs : int
        Number of input nodes (flattened agent output + env signals)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    prediction_size : int
        Size of the target vectors `(ŷ, ẑ)` and policy hidden representation
    max_action_dim : int
        Maximum continuous action dimensionality across all environments.
        Determines the output width of `μ̂` and `log σ̂` projections
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
        max_action_dim: int,
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
    ) -> None:
        self.pred_size = prediction_size
        self.max_action_dim = max_action_dim

        super().__init__(
            n_inputs,
            n_neurons,
            key=key,
            sparsity=sparsity,
        )

        # Gaussian policy target projections: pi_hidden -> (μ̂, log σ̂)
        self.mu_target_proj = nnx.Linear(
            prediction_size,
            max_action_dim,
            rngs=self.rngs,
        )
        self.log_std_target_proj = nnx.Linear(
            prediction_size,
            max_action_dim,
            rngs=self.rngs,
        )

    def _build_wiring(self) -> NCPWiringSpec:
        return build_ncp_wiring(
            self.in_features,
            self.n_neurons,
            DiscoHeadSpec,
            seed=self.seed,
            sparsity=self.sparsity,
            pi=self.pred_size,
            y=self.pred_size,
            z=self.pred_size,
        )

    def __call__(
        self,
        obs: jax.Array,
        *,
        h_state: Optional[jax.Array] = None,
        timespans: Optional[jax.Array] = None,
    ) -> Tuple[DiscoPredictions, jax.Array]:
        """
        Forward pass through the network.

        Parameters
        ----------
        obs : jax.Array
            Input observations with shape `(B, T, F)` or `(B, F)`

            - `batch_size (B)`: the number of samples per timestep
            - `seq_length (T)`: the number of sequences (e.g., trajectories)
            - `features (F)`: the features at each timestep
        h_state : jax.Array (optional)
            Initial hidden state with shape `(B, H)`

            - `batch_size (B)`: the number of samples per timestep
            - `n_hidden (H)`: the total number of hidden neurons
        timespans : jax.Array (optional)
            Time elapsed since previous timestep. For fixed intervals set to `None`.
            For varying timesteps shape must be `(T,)`

            - `seq_length (T)`: the number of sequences (e.g., trajectories)

        Returns
        -------
        target_preds : DiscoPredictions
            Network predictions for the command layer (`embedding`) and each
            head `(μ̂, log σ̂, ŷ, ẑ)`
        h_state : jax.Array
            Final hidden state with shape `(B, H)`
        """
        x, h_state, timespans = self._preprocess(obs, h_state, timespans)
        h_state, preds = self._scan(x, h_state, timespans, reverse=True)

        # 4 outputs -> (embedding, pi_hidden, ŷ, ẑ)
        embedding, pi_hidden, y, z = self._postprocess(preds)

        # Project pi_hidden to Gaussian target parameters
        mu = self.mu_target_proj(pi_hidden)  # type: ignore
        log_std = self.log_std_target_proj(pi_hidden)  # type: ignore

        return DiscoPredictions(
            embedding=embedding,
            mu=mu,
            log_std=log_std,
            y=y,
            z=z,
        ), h_state
