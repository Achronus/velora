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

from typing import Tuple

import torch
from torch import nn

from velora.lnn.backbone import CfCBackbone


class LNN(nn.Module):
    """
    A CfC Liquid Neural Circuit Policy (NCP) Network with three layers.

    Layers -
        1. Inter (input) - a `AdaptiveLiquidCell` layer
        2. Command (hidden) - a `AdaptiveLiquidCell` layer
        3. Motor (output) - a `AdaptiveLiquidCell` layer

    ??? note "Decision nodes"

        `inter` and `command` neurons are automatically calculated using:

        ```python
        command_neurons = max(int(0.4 * n_neurons), 1)
        inter_neurons = n_neurons - command_neurons
        ```

    Combines a Liquid Time-Constant (LTC) cell with Ordinary Neural Circuits (ONCs).

    References -
        - [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898)
        - [Reinforcement Learning with Ordinary Neural Circuits](https://proceedings.mlr.press/v119/hasani20a.html)

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter and command nodes)
    out_features : int
        Number of out features (motor nodes)
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
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        seed: int = 28,
        sparsity: float = 0.5,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features
        self.seed = seed
        self.sparsity = sparsity
        self.alpha_rank = alpha_rank

        self.backbone = CfCBackbone(
            in_features,
            n_neurons,
            heads={"out": self.out_features},
            seed=seed,
            sparsity=sparsity,
            alpha_rank=alpha_rank,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            An input array of shape: `(B, F)` or `(B, T, F)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories, channels)
            - `features (F)` the features at each timestep

        h : torch.Tensor (optional)
            Initial hidden state of the RNN with shape: `(B, H)`

            - `batch_size (B)` the number of samples per timestep
            - `n_hidden (H)` the total number of hidden neurons

        ts : torch.Tensor (optional)
            Time elapsed since previous timestep.
            For fixed intervals set to `None`. For varying timesteps shape
            should be `(T,)`

        Returns
        -------
        out_preds : torch.Tensor
            The network prediction. Shape `(B, T, F)`
        h_state : torch.Tensor
            The final hidden state. Shape `(B, H)`
        """
        preds, h_state = self.backbone(x, h=h, ts=ts)
        return preds["out"], h_state
