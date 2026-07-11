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

import torch
from torch import nn

from velora.nn.lnn.wiring import build_layer_mask
from velora.nn.sparse import SparseLinear
from velora.utils.nn import active_parameters, total_parameters


class SparseGatedBlock(nn.Module):
    """
    A stateless sparse gated block.

    Combines NCP-style sparse wiring with the CfC-style multiplicative
    gate from `AdaptiveLiquidCell`, with the timescale, decay, erasure,
    and recurrence mechanisms removed. A drop-in replacement for a
    `Linear -> activation` block.

    Equation:
    $$
    y = g(x, θ_g) \\; (1 - σ(f(x, θ_f))) + h(x, θ_h) \\; σ(f(x, θ_f))
    $$

    Parameters
    ----------
    in_features : int
        Number of inputs nodes
    out_features : int
        Number of output nodes
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Default is `0.5`.
        Must be a value between `[0.1, 0.9]`:

        - Where `0.1` neurons are very dense
        - Where `0.9` neurons are very sparse
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        seed: int = 28,
        sparsity: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        mask = torch.abs(
            build_layer_mask(
                in_features,
                out_features,
                seed=seed,
                sparsity=sparsity,
            )
        )

        self.g_head = SparseLinear(self.in_features, self.out_features, mask)
        self.h_head = SparseLinear(self.in_features, self.out_features, mask)
        self.f_head = SparseLinear(self.in_features, self.out_features, mask)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns
        -------
        count : int
            The total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns
        -------
        count : int
            The active parameter count.
        """
        return self._active_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the gated block.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of shape `(B, F)` or `(B, T, F)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories)
            - `features (F)` the features at each timestep

        Returns
        -------
        preds : torch.Tensor
            Block predictions of shape `(B, T, F_out)`
        """
        g = self.tanh(self.g_head(x))
        h = self.tanh(self.h_head(x))
        gate = self.sigmoid(self.f_head(x))

        return g * (1.0 - gate) + h * gate
