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


class ACMPredictions(NamedTuple):
    """
    Output storage for `ACM` network predictions.

    Parameters
    ----------
    embedding : torch.Tensor
        Command layer output with shape `(B, T, F)`

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_features (`F`) - the number of features.
    z : torch.Tensor
        Action-conditioned prediction vector with shape `(B, T, Z)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - z_dim (`Z`) - the size of the action-conditioned prediction vector (prediction_size)
    aux_pi : torch.Tensor
        Auxiliary policy prediction with shape `(B, T, H)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - hidden_dim (`H`) - the auxiliary policy hidden representation size (prediction_size)
    q : torch.Tensor
        Scalar Q-value predictions with shape `(B, T, 1)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
    """

    embedding: torch.Tensor
    z: torch.Tensor
    aux_pi: torch.Tensor
    q: torch.Tensor


class ACM(nn.Module):
    """
    Action-Conditional Model (ACM) for continuous action spaces.

    Takes the OCM state embedding concatenated with the continuous action
    vector as input, producing action-conditioned predictions `z(s, a)`,
    auxiliary policy predictions `aux_pi(s, a)`, and scalar Q-values `q(s, a)`.

    Uses a CfC-LNN core with 3 output heads:

        1. Action-conditioned prediction: `z(s, a)` — learned representation
           with discovered semantics
        2. Auxiliary policy prediction: `aux_pi(s, a)` — predicts next-step
           policy parameters for representation learning
        3. Action-value: `q(s, a)` — scalar Q-value for the taken action

    Hidden state layout: `[core (inter, command), z, aux_pi, q]`.

    Parameters
    ----------
    obs_dim : int
        Number of observations (OCM embedding size)
    max_action_dim : int
        Maximum continuous action dimensionality across all environments.
        Actions are zero-padded to this size for static input shapes.
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    prediction_size : int
        Size of the action-conditioned prediction vector `z` and
        auxiliary policy hidden representation
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
        max_action_dim: int,
        n_neurons: int,
        prediction_size: int,
        *,
        seed: int = 28,
        sparsity: float = 0.5,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.max_action_dim = max_action_dim
        self.n_neurons = n_neurons
        self.z_dim = prediction_size
        self.aux_pi_dim = prediction_size
        self.q_dim = 1  # Scalar Q-value
        self.sparsity = sparsity

        # Input: OCM embedding + continuous action vector
        in_features = obs_dim + max_action_dim

        wiring = build_wiring(
            in_features,
            n_neurons,
            {"z": self.z_dim, "aux_pi": self.aux_pi_dim, "q": self.q_dim},
            seed=seed,
            sparsity=sparsity,
        )  # masks = (out, in)

        self.core = CfCCore(wiring.inter, wiring.command, alpha_rank=alpha_rank)

        self.z_head = AdaptiveLiquidCell(
            self.core.command_size,
            self.z_dim,
            wiring.heads["z"],
            alpha_rank=alpha_rank,
        )
        self.aux_pi_head = AdaptiveLiquidCell(
            self.core.command_size,
            self.aux_pi_dim,
            wiring.heads["aux_pi"],
            alpha_rank=alpha_rank,
        )
        self.q_head = AdaptiveLiquidCell(
            self.core.command_size,
            self.q_dim,
            wiring.heads["q"],
            alpha_rank=alpha_rank,
        )

        self.embedding_size = self.core.command_size

        self.hidden_sizes = [
            self.core.hidden_size,
            self.z_dim,
            self.aux_pi_dim,
            self.q_dim,
        ]
        self.hidden_size = sum(self.hidden_sizes)

        self.total_params = total_parameters(self)
        self.active_params = active_parameters(self)

    def forward(
        self,
        state_embedding: torch.Tensor,
        action: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> Tuple[ACMPredictions, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Concatenates the state embedding with the continuous action vector
        before processing through the core and heads.

        Parameters
        ----------
        state_embedding : torch.Tensor
            Embedded state from OCM with shape `(B, T, F)` or `(B, F)`
        action : torch.Tensor
            Continuous action vector with shape `(B, T, D)` or `(B, D)`.
            Padded to `max_action_dim`.
        h : torch.Tensor (optional)
            Initial hidden state with shape `(B, H)`
        ts : torch.Tensor (optional)
            Time elapsed since previous timestep `(T,)`

        Returns
        -------
        acm_preds : ACMPredictions
            Network predictions `(embedding, z, aux_pi, q)`
        h_state : torch.Tensor
            Final hidden state with shape `(B, H)`
        """
        x = torch.cat([state_embedding, action], dim=-1)

        x, h, ts = prepare_inputs(x, h, ts, self.hidden_size)
        h_core, h_z, h_aux_pi, h_q = h.split(self.hidden_sizes, dim=1)

        embedding, h_core = self.core(x, h_core, ts)

        z, h_z = scan_head(self.z_head, embedding, h_z, ts)
        aux_pi, h_aux_pi = scan_head(self.aux_pi_head, embedding, h_aux_pi, ts)
        q, h_q = scan_head(self.q_head, embedding, h_q, ts)

        h_state = torch.cat([h_core, h_z, h_aux_pi, h_q], dim=1)
        return ACMPredictions(embedding=embedding, z=z, aux_pi=aux_pi, q=q), h_state
