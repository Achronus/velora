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

from velora.nn.lnn.cell import AdaptiveLiquidCell
from velora.nn.lnn.wiring import build_wiring
from velora.utils.nn import active_parameters, total_parameters


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
        Number of decision nodes (inter + command nodes)
    out_features : int
        Number of output features
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
    init_std : float (optional)
        Gain for orthogonal weight initialization (e.g., `np.sqrt(2)`).
        When `None`, uses Kaiming uniform initialization instead (PyTorch default).
        Default is `None`
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
        init_std: float | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features
        self.sparsity = sparsity

        wiring = build_wiring(
            in_features,
            n_neurons,
            heads={"out": out_features},
            seed=seed,
            sparsity=sparsity,
        )  # masks = (out, in)

        # Inter layer: sensory -> inter
        self.inter = AdaptiveLiquidCell(
            self.in_features,
            wiring.inter.shape[0],
            wiring.inter,
            alpha_rank=alpha_rank,
            init_std=init_std,
        )

        # Command layer: inter -> command
        self.command = AdaptiveLiquidCell(
            wiring.inter.shape[0],
            wiring.command.shape[0],
            wiring.command,
            alpha_rank=alpha_rank,
            init_std=init_std,
        )

        # Motor heads: command -> motors (outputs)
        self.motor = AdaptiveLiquidCell(
            wiring.command.shape[0],
            wiring.heads["out"].shape[0],
            wiring.heads["out"],
            alpha_rank=alpha_rank,
            init_std=init_std,
        )

        self.hidden_sizes = [
            wiring.inter.shape[0],
            wiring.command.shape[0],
            wiring.heads["out"].shape[0],
        ]
        self.hidden_size = sum(self.hidden_sizes)

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def command_size(self) -> int:
        """
        Gets the command layer's neuron count (the `embedding` width).

        Returns
        -------
        count : int
            The number of command neurons
        """
        return self.hidden_sizes[1]

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

    def _preprocess(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Utility method that prepares `forward` inputs.

        Includes -
        - `x` dimension expansion from `(B, F)` -> `(B, T, F)` (if needed)
        - `h` initialized to zeros of shape `(B, H)` when set to `None`
        - `ts` initialized to ones of shape `(T,)` when set to `None`

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of shape `(B, F)` or `(B, T, F)`
        h : torch.Tensor (optional)
            Initial hidden state of shape `(B, H)`
        ts : torch.Tensor (optional)
            Time elapsed since previous timestep.
            For fixed intervals set to `None`. For varying timesteps
            shape should be `(T,)`

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
            h = torch.zeros((B, self.hidden_size), device=x.device, dtype=x.dtype)

        ts = torch.ones(T, device=x.device, dtype=x.dtype) if ts is None else ts

        return x, h, ts

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network, one timestep at a time.

        At each timestep, steps the inter and command layers in sequence,
        then each motor head on the command layer's output (embedding).
        Hidden states are threaded through the scan in the layout
        `[inter, command, *heads]` along the feature dimension.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of shape `(B, F)` or `(B, T, F)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories)
            - `features (F)` the features at each timestep

        h : torch.Tensor (optional)
            Initial hidden state of shape `(B, H)`.
            Initialized to zeros when `None`

            - `n_hidden (H)` the total number of hidden neurons

        ts : torch.Tensor (optional)
            Time elapsed since previous timestep.
            For fixed intervals set to `None`. For varying timesteps
            shape should be `(T,)`

        Returns
        -------
        preds : torch.Tensor
            Network predictions of shape `(B, T, F_out)`
        h_state : torch.Tensor
            The final hidden state of shape `(B, H)`
        """
        x, h, ts = self._preprocess(x, h, ts)

        # Split hidden for each layer
        h_inter, h_command, h_motor = h.split(self.hidden_sizes, dim=1)

        # Iterate over T dim
        y_preds = []
        for t in range(x.shape[1]):
            x_t, ts_t = x[:, t], ts[t]

            x_t, h_inter = self.inter(x_t, h_inter, ts_t)
            embed_t, h_command = self.command(x_t, h_command, ts_t)
            y_t, h_motor = self.motor(embed_t, h_motor, ts_t)
            y_preds.append(y_t)

        h_state = torch.cat([h_inter, h_command, h_motor], dim=1)
        y_preds = torch.stack(y_preds, dim=1)

        return y_preds, h_state
