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
import torch.nn.functional as F
from torch import nn


class SparseLinear(nn.Module):
    """
    A linear layer with sparsely weighted connections.

    Equation:
    $$
    y = x * (w * m) + b
    $$

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    mask : torch.Tensor
        Sparsity mask (m) tensor of shape
        `(out_features, in_features)`
    """

    mask: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        weights = torch.nn.init.kaiming_uniform_(
            torch.empty((out_features, in_features)),
            nonlinearity="linear",
        )

        self.register_buffer("mask", mask)

        self.weights = nn.Parameter(weights * self.mask)
        self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a linear transformation to the inputs along the last dimension.

        Parameters
        ----------
        x : torch.Tensor
            The array to transform with shape `(..., in_features)`

        Returns
        -------
        y_pred : torch.Tensor
            The layer prediction with sparsity applied. Has shape `(..., out_features)`
        """
        return F.linear(x, self.weights * self.mask, self.bias)
