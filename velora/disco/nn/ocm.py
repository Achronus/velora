# Copyright 2026 Achronus
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

from typing import NamedTuple, Tuple

import torch
from torch import nn

from velora.lnn.ncp import LNN


class OCMPredictions(NamedTuple):
    """
    Output storage for `OCM` network predictions.

    Parameters
    ----------
    embedding : torch.Tensor
        Command layer output with shape `(B, T, F)`

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_features (`F`) - the number of features.
    pi : torch.Tensor
        Policy logits with shape `(B, T, H)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - hidden_dim (`H`) - the policy hidden representation size (prediction_size)
    y : torch.Tensor
        Observation-conditioned prediction vector with shape `(B, T, Y)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - y_dim (`Y`) - the size of the observation-conditioned prediction vector.
    """

    embedding: torch.Tensor
    pi: torch.Tensor
    y: torch.Tensor


class OCM(nn.Module):
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
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Default is `0.5`.

        Must be a value between `[0.1, 0.9]`:

            - Where `0.1` neurons are very dense
            - Where `0.9` neurons are very sparse
    alpha_rank : int (optional)
        Rank of the low-rank α projection. Default is `min(n_hidden, 4)`
    """

    def __init__(
        self,
        obs_dim: int,
        n_neurons: int,
        prediction_size: int,
        *,
        seed: int = 28,
        sparsity: float = 0.5,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.n_neurons = n_neurons
        self.y_dim = prediction_size
        self.pi_hidden_dim = prediction_size
        self.sparsity = sparsity

        self.lnn = LNN(
            obs_dim,
            n_neurons,
            heads={"pi": self.pi_hidden_dim, "y": self.y_dim},
            seed=seed,
            sparsity=sparsity,
            alpha_rank=alpha_rank,
        )

        self.embedding_size = self.lnn.command_size

    def forward(
        self,
        obs: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> Tuple[OCMPredictions, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        obs : torch.Tensor
            Input observations with shape `(B, T, F)` or `(B, F)`

            - `batch_size (B)`: the number of samples per timestep
            - `seq_length (T)`: the number of sequences (e.g., trajectories)
            - `features (F)`: the features at each timestep
        h : torch.Tensor (optional)
            Initial hidden state with shape `(B, H)`

            - `batch_size (B)`: the number of samples per timestep
            - `n_hidden (H)`: the total number of hidden neurons
        ts : torch.Tensor (optional)
            Time elapsed since previous timestep. For fixed intervals set to `None`.
            For varying timesteps shape must be `(T,)`

            - `seq_length (T)`: the number of sequences (e.g., trajectories)

        Returns
        -------
        ocm_preds : OCMPredictions
            Network predictions for the command layer (`embedding`)
            and each head `(pi, y)`
        h : torch.Tensor
            Final hidden state with shape `(B, H)`
        """
        preds, h_state = self.lnn(obs, h=h, ts=ts)
        return OCMPredictions(preds["embedding"], preds["pi"], preds["y"]), h_state
