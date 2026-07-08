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

from velora.disco.nn.cfc import CfCCore, prepare_inputs, scan_head
from velora.lnn.cell import AdaptiveLiquidCell
from velora.lnn.wiring import build_wiring
from velora.utils.nn import active_parameters, total_parameters


class DiscoPredictions(NamedTuple):
    """
    Output storage for `DiscoNetwork` predictions.

    Parameters
    ----------
    embedding : torch.Tensor
        Command layer output with shape `(B, T, F)`

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_features (`F`) - the number of features.
    mu : torch.Tensor
        Gaussian policy target means `μ̂` with shape `(B, T, D)`:

        - action_dim (`D`) - the maximum continuous action dimensionality (max_action_dim)
    log_std : torch.Tensor
        Gaussian policy target log standard deviations `log σ̂` with
        shape `(B, T, D)`:

        - action_dim (`D`) - the maximum continuous action dimensionality (max_action_dim)
    y : torch.Tensor
        Observation-conditioned targets `ŷ` with shape `(B, T, Y)`:

        - y_dim (`Y`) - the size of the target vector (prediction_size)
    z : torch.Tensor
        Action-conditioned targets `ẑ` with shape `(B, T, Z)`:

        - z_dim (`Z`) - the size of the target vector (prediction_size)
    """

    embedding: torch.Tensor
    mu: torch.Tensor
    log_std: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor


class DiscoNetwork(nn.Module):
    """
    Meta-network that produces learned targets `(μ̂, log σ̂, ŷ, ẑ)` for the
    update rule. Suitable for continuous action spaces.

    Uses a CfC-LNN core with action-agnostic output heads:

        1. Policy target predictions: `μ̂(s, a)`, `log σ̂(s, a)`
        2. Observation-conditioned targets: `ŷ(s)`
        3. Action-conditioned targets: `ẑ(s, a)`

    Processes trajectories back-to-front (bootstrapping): predictions at
    each timestep are conditioned on the trajectory's future, not its past.
    Predictions are returned in forward order.

    Hidden state layout: `[core (inter, command), pi, y, z]`.

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
        n_inputs: int,
        n_neurons: int,
        prediction_size: int,
        max_action_dim: int,
        *,
        seed: int = 28,
        sparsity: float = 0.5,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__()

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.pred_size = prediction_size
        self.max_action_dim = max_action_dim
        self.sparsity = sparsity

        wiring = build_wiring(
            n_inputs,
            n_neurons,
            {"pi": prediction_size, "y": prediction_size, "z": prediction_size},
            seed=seed,
            sparsity=sparsity,
        )  # masks = (out, in)

        self.core = CfCCore(wiring.inter, wiring.command, alpha_rank=alpha_rank)

        self.pi_head = AdaptiveLiquidCell(
            self.core.command_size,
            prediction_size,
            wiring.heads["pi"],
            alpha_rank=alpha_rank,
        )
        self.y_head = AdaptiveLiquidCell(
            self.core.command_size,
            prediction_size,
            wiring.heads["y"],
            alpha_rank=alpha_rank,
        )
        self.z_head = AdaptiveLiquidCell(
            self.core.command_size,
            prediction_size,
            wiring.heads["z"],
            alpha_rank=alpha_rank,
        )

        # Gaussian policy target projections: pi_hidden -> (μ̂, log σ̂)
        self.mu_target_proj = nn.Linear(prediction_size, max_action_dim)
        self.log_std_target_proj = nn.Linear(prediction_size, max_action_dim)

        self.embedding_size = self.core.command_size

        self.hidden_sizes = [
            self.core.hidden_size,
            prediction_size,
            prediction_size,
            prediction_size,
        ]
        self.hidden_size = sum(self.hidden_sizes)

        self.total_params = total_parameters(self)
        self.active_params = active_parameters(self)

    def forward(
        self,
        obs: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> Tuple[DiscoPredictions, torch.Tensor]:
        """
        Forward pass through the network, processing the sequence in reverse.

        Inputs are flipped along the time dimension before scanning, so the
        recurrence runs back-to-front. Predictions are flipped back to
        forward order before returning.

        Parameters
        ----------
        obs : torch.Tensor
            Input observations with shape `(B, T, F)` or `(B, F)`

            - `batch_size (B)`: the number of samples per timestep
            - `seq_length (T)`: the number of sequences (e.g., trajectories)
            - `features (F)`: the features at each timestep
        h : torch.Tensor (optional)
            Initial hidden state with shape `(B, H)`, applied at the
            *end* of the trajectory (the reverse scan's starting point)

            - `n_hidden (H)`: the total number of hidden neurons
        ts : torch.Tensor (optional)
            Time elapsed since previous timestep. For fixed intervals set to `None`.
            For varying timesteps shape must be `(T,)`

        Returns
        -------
        target_preds : DiscoPredictions
            Network predictions for the command layer (`embedding`) and each
            head `(μ̂, log σ̂, ŷ, ẑ)`, in forward time order
        h_state : torch.Tensor
            Final hidden state with shape `(B, H)`
        """
        obs, h, ts = prepare_inputs(obs, h, ts, self.hidden_size)

        # Bootstrapping: scan back-to-front
        obs = obs.flip(1)
        ts = ts.flip(0)

        h_core, h_pi, h_y, h_z = h.split(self.hidden_sizes, dim=1)

        embedding, h_core = self.core(obs, h_core, ts)

        pi_hidden, h_pi = scan_head(self.pi_head, embedding, h_pi, ts)
        y, h_y = scan_head(self.y_head, embedding, h_y, ts)
        z, h_z = scan_head(self.z_head, embedding, h_z, ts)

        # Restore forward time order
        embedding = embedding.flip(1)
        pi_hidden = pi_hidden.flip(1)
        y = y.flip(1)
        z = z.flip(1)

        # Project pi_hidden to Gaussian target parameters
        mu = self.mu_target_proj(pi_hidden)
        log_std = self.log_std_target_proj(pi_hidden)

        h_state = torch.cat([h_core, h_pi, h_y, h_z], dim=1)
        return DiscoPredictions(
            embedding=embedding,
            mu=mu,
            log_std=log_std,
            y=y,
            z=z,
        ), h_state
