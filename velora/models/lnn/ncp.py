from collections import OrderedDict
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn

from velora.models.lnn.cell import NCPLiquidCell
from velora.models.lnn.wiring import Wiring


class LiquidNCPNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        sparsity_level: float = 0.5,
        seed: int = 64,
    ) -> None:
        """
        A Liquid Neural Circuit Policy network with three layers:
        1. Inter (includes sensory inputs)
        2. Command
        3. Motor (output)

        Parameters:
            in_features (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes).
                Nodes are set automatically based on the following:
                ```python
                command_neurons = max(int(0.4 * n_neurons), 1)
                inter_neurons = n_neurons - command_neurons
                ```
            out_features (int): number of out features (motor nodes)
            sparsity_level (float, optional): controls the connection sparsity between
                neurons. Must be a value between `[0.1, 0.9]`. When `0.1` neurons are
                very dense, when `0.9` they are very sparse. Default is '0.5'
            seed (int, optional): random seed for reproducibility. Default is '64'
        """
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features

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
            ),
            NCPLiquidCell(
                self._counts.inter,
                self._counts.command,
                self._masks.command,
            ),
            NCPLiquidCell(
                self._counts.command,
                self._counts.motor,
                self._masks.motor,
            ),
        ]
        self.layers = OrderedDict([(name, layer) for name, layer in zip(names, layers)])

        self.ncp = nn.Sequential(self.layers)
        self._out_sizes = [layer.n_hidden for layer in self.layers.values()]

    def _ncp_forward(
        self, x: torch.Tensor, hidden: torch.Tensor, timespans: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the NCP network.

        Splits the hidden state into respective chunks (out_features) for each layer
        to maintain their own independent hidden state dynamics. Then, merges them
        together to create a new hidden state.

        Parameters:
            x (torch.Tensor): an input tensor
            hidden (torch.Tensor): the hidden states for all layers
            timespans (torch.Tensor | float): a single or set of time
                intervals between events

        Returns:
            h,new_h_state (Tuple[torch.Tensor, torch.Tensor]): current hidden
            state (network prediction) and merged hidden state from all
            layers (updated state memory).
        """
        h_state = torch.split(hidden, self._out_sizes, dim=1)

        new_h_state = []

        # Handle layer independence
        for i, layer in enumerate(self.layers.values()):
            h = layer.forward(x, h_state[i], timespans)
            x = h
            new_h_state.append(h)

        new_h_state = torch.cat(new_h_state, dim=1)
        return h, new_h_state

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            x (torch.Tensor): an input tensor of shape: `(batch, seq, channels)`
            hidden (torch.Tensor, optional): initial hidden state of the RNN with shape: `(batch, hidden)`. Default is 'None'
            timespans (torch.Tensor, optional): a single or set of time
                intervals between events. Used for event-based data. When `None`
                is set to `1.0`. Default is 'None'

        Returns:
            y_pred,h_state (Tuple[torch.Tensor, torch.Tensor]): the network prediction and the final hidden state.
        """
        if x.dim() == 3:
            batch_size, seq_len, channels = x.size()
        else:
            batch_size, seq_len = x.size()

        if hidden is None:
            hidden = torch.zeros((batch_size, self.n_neurons))

        # Add batch dimension
        if timespans is not None:
            timespans = timespans.unsqueeze(0)

        output_sequence = []
        for t in range(seq_len):
            inputs = x[:, t]
            ts = 1.0 if timespans is None else timespans[:, t].squeeze()

            h_out, h_state = self._ncp_forward(inputs, hidden, ts)
            output_sequence.append(h_out)

        y_pred = torch.stack(output_sequence, dim=1)
        return y_pred, h_state
