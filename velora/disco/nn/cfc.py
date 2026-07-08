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

from velora.lnn.cell import AdaptiveLiquidCell


def prepare_inputs(
    x: torch.Tensor,
    h: torch.Tensor | None,
    ts: torch.Tensor | None,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Utility method that prepares network `forward` inputs.

    Includes -
    - `x` dimension expansion from `(B, F)` -> `(B, T, F)` (if needed)
    - `h` initialized to zeros of shape `(B, H)` when set to `None`
    - `ts` initialized to ones of shape `(T,)` when set to `None`

    Parameters
    ----------
    x : torch.Tensor
        An input tensor of shape `(B, F)` or `(B, T, F)`
    h : torch.Tensor
        Initial hidden state of shape `(B, H)`, or `None`
    ts : torch.Tensor
        Time elapsed since previous timestep of shape `(T,)`, or `None`
    hidden_size : int
        The network's total hidden state size (`H`)

    Returns
    -------
    x : torch.Tensor
        Prepared input of shape `(B, T, F)`
    h : torch.Tensor
        Hidden state of shape `(B, H)`
    ts : torch.Tensor
        Time intervals of shape `(T,)`
    """
    if x.ndim == 2:
        x = x.unsqueeze(dim=1)

    B, T, _ = x.shape

    if h is None:
        h = torch.zeros((B, hidden_size), device=x.device, dtype=x.dtype)

    ts = torch.ones(T, device=x.device, dtype=x.dtype) if ts is None else ts

    return x, h, ts


def scan_head(
    cell: nn.Module,
    embeds: torch.Tensor,
    h: torch.Tensor,
    ts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scans a motor head cell over a sequence of embeddings.

    Parameters
    ----------
    cell : nn.Module
        A liquid cell head
    embeds : torch.Tensor
        Command layer outputs of shape `(B, T, C)`
    h : torch.Tensor
        The head's initial hidden state of shape `(B, n_hidden)`
    ts : torch.Tensor
        Time elapsed since previous timestep of shape `(T,)`

    Returns
    -------
    preds : torch.Tensor
        Head predictions of shape `(B, T, n_hidden)`
    h : torch.Tensor
        The head's final hidden state of shape `(B, n_hidden)`
    """
    outs = []

    for t in range(embeds.shape[1]):
        out_t, h = cell(embeds[:, t], h, ts[t])
        outs.append(out_t)

    return torch.stack(outs, dim=1), h


class CfCCore(nn.Module):
    """
    The core layers of a CfC-LNN NCP-based network.

    Runs the sequential recurrence over the inter and command layers,
    producing the command embeddings that motor heads consume. Motor
    heads have no feedback into the core, so networks scan them
    separately over the returned embeddings (see `scan_head`).

    Layers -
        1. Inter (input) - a `AdaptiveLiquidCell` layer
        2. Command (hidden) - a `AdaptiveLiquidCell` layer

    References -
        - [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898)
        - [Reinforcement Learning with Ordinary Neural Circuits](https://proceedings.mlr.press/v119/hasani20a.html)

    Parameters
    ----------
    inter_mask : torch.Tensor
        Inter layer sparsity mask of shape `(n_inter, in_features)`
    command_mask : torch.Tensor
        Command layer sparsity mask of shape `(n_command, n_inter)`
    alpha_rank : int (optional)
        Rank of the low-rank α projection. Default is `min(n_hidden, 4)`
    """

    def __init__(
        self,
        inter_mask: torch.Tensor,
        command_mask: torch.Tensor,
        *,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__()

        self.in_features = inter_mask.shape[1]
        self.command_size = command_mask.shape[0]

        self.hidden_sizes = [inter_mask.shape[0], command_mask.shape[0]]
        self.hidden_size = sum(self.hidden_sizes)

        self.inter = AdaptiveLiquidCell(
            self.in_features,
            inter_mask.shape[0],
            inter_mask,
            alpha_rank=alpha_rank,
        )
        self.command = AdaptiveLiquidCell(
            inter_mask.shape[0],
            command_mask.shape[0],
            command_mask,
            alpha_rank=alpha_rank,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        ts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scans the inter and command layers over the input sequence.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of shape `(B, T, F)`
        h : torch.Tensor
            The core's initial hidden state of shape `(B, n_inter + n_command)`,
            laid out as `[inter, command]`
        ts : torch.Tensor
            Time elapsed since previous timestep of shape `(T,)`

        Returns
        -------
        embeds : torch.Tensor
            Command layer outputs (embeddings) of shape `(B, T, n_command)`
        h : torch.Tensor
            The core's final hidden state of shape `(B, n_inter + n_command)`
        """
        h_inter, h_command = h.split(self.hidden_sizes, dim=1)

        embeds = []

        for t in range(x.shape[1]):
            x_t, h_inter = self.inter(x[:, t], h_inter, ts[t])
            embed_t, h_command = self.command(x_t, h_command, ts[t])
            embeds.append(embed_t)

        return torch.stack(embeds, dim=1), torch.cat([h_inter, h_command], dim=1)
