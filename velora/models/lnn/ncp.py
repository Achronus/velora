from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from velora.models.lnn.cell import NCPLiquidCell
from velora.wiring import Wiring
from velora.utils.torch import active_parameters, total_parameters


class LiquidNCPNetwork(nn.Module):
    """
    A Liquid Neural Circuit Policy (NCP) Network with three layers:

    1. Inter (input)
    2. Command (hidden)
    3. Motor (output)

    Each layer is a `NCPLiquidCell`.

    !!! note "Decision nodes"

        `inter` and `command` neurons are automatically calculated using:

        ```python
        command_neurons = max(int(0.4 * n_neurons), 1)
        inter_neurons = n_neurons - command_neurons
        ```
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        sparsity_level: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            in_features (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes).
            out_features (int): number of out features (motor nodes)
            sparsity_level (float, optional): controls the connection sparsity
                between neurons.

                Must be a value between `[0.1, 0.9]` -

                - When `0.1` neurons are very dense.
                - When `0.9` they are very sparse.

            device (torch.device, optional): the device to load tensors on.
        """
        super().__init__()

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features
        self.device = device

        self.n_units = n_neurons + out_features  # inter + command + motor

        self._wiring = Wiring(
            in_features,
            n_neurons,
            out_features,
            sparsity_level=sparsity_level,
        )
        self._masks, self._counts = self._wiring.data()

        names = ["inter", "command", "motor"]
        layers = [
            NCPLiquidCell(
                in_features,
                self._counts.inter,
                self._masks.inter,
            ).to(device),
            NCPLiquidCell(
                self._counts.inter,
                self._counts.command,
                self._masks.command,
            ).to(device),
            NCPLiquidCell(
                self._counts.command,
                self._counts.motor,
                self._masks.motor,
            ).to(device),
        ]
        self.layers = OrderedDict([(name, layer) for name, layer in zip(names, layers)])

        self.ncp = nn.Sequential(self.layers)
        self._out_sizes = [layer.n_hidden for layer in self.layers.values()]

        self._total_params = total_parameters(self.ncp)
        self._active_params = active_parameters(self.ncp)

    @property
    def total_params(self) -> int:
        """
        Returns the network's total parameter count.

        Returns:
            count (int): the total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Returns the network's activate parameter count.

        Returns:
            count (int): the active parameter count.
        """
        return self._active_params

    def _ncp_forward(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single timestep through the network layers.

        Splits the hidden state into respective chunks for each layer
        (`out_features`) to maintain their own independent hidden state dynamics.

        Then, merges them together to create a new hidden state.

        Parameters:
            x (torch.Tensor): the current batch of data for the timestep with
                shape: `(batch_size, features)`
            hidden (torch.Tensor): the current hidden state

        Returns:
            y_pred (torch.Tensor): the network prediction.
            new_h_state (torch.Tensor): the merged hidden state from all layers (updated state memory).
        """
        h_state = torch.split(hidden, self._out_sizes, dim=1)

        new_h_state = []
        inputs = x

        # Handle layer independence
        for i, layer in enumerate(self.layers.values()):
            y_pred, h = layer(inputs, h_state[i])
            inputs = y_pred  # (batch_size, layer_out_features)
            new_h_state.append(h)

        new_h_state = torch.cat(new_h_state, dim=1)  # (batch_size, n_units)
        return y_pred, new_h_state

    def forward(
        self, x: torch.Tensor, h_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            x (torch.Tensor): an input tensor of shape: `(batch_size, features)`.

                - `batch_size` the number of samples per timestep.
                - `features` the features at each timestep (e.g.,
                image features, joint coordinates, word embeddings, raw amplitude
                values).
            h_state (torch.Tensor, optional): initial hidden state of the RNN with
                shape: `(batch_size, n_units)`.

                - `batch_size` the number of samples.
                - `n_units` the total number of hidden neurons
                    (`n_neurons + out_features`).

        Returns:
            y_pred (torch.Tensor): the network prediction. When `batch_size=1`. Out shape is `(out_features)`. Otherwise, `(batch_size, out_features)`.
            h_state (torch.Tensor): the final hidden state. Output shape is `(batch_size, n_units)`.
        """
        if x.dim() != 2:
            raise ValueError(
                f"Unsupported dimensionality: '{x.shape=}'. Should be 2 dimensional with: '(batch_size, features)'."
            )

        x = x.to(torch.float32).to(self.device)

        batch_size, features = x.size()

        if h_state is None:
            h_state = torch.zeros((batch_size, self.n_units), device=self.device)

        # Batch -> (batch_size, out_features)
        y_pred, h_state = self._ncp_forward(x, h_state.to(self.device))

        # Single item -> (out_features)
        if y_pred.shape[0] == 1:
            y_pred = y_pred.squeeze(0)

        # h_state -> (batch_size, n_units)
        return y_pred, h_state
