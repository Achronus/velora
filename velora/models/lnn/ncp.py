from typing import Optional, Tuple

import torch
import torch.nn as nn

from velora.models.lnn.cell import NCPLiquidCell
from velora.models.lnn.sparse import SparseLinear
from velora.models.weight import WeightInitType
from velora.utils.torch import active_parameters, total_parameters
from velora.wiring import Wiring


class LiquidNCPNetwork(nn.Module):
    """
    A CfC Liquid Neural Circuit Policy (NCP) Network with three layers:

    1. Inter (input) - a `SparseLinear` layer
    2. Command (hidden) - a `NCPLiquidCell` layer
    3. Motor (output) - a `SparseLinear` layer

    ??? note "Decision nodes"

        `inter` and `command` neurons are automatically calculated using:

        ```python
        command_neurons = max(int(0.4 * n_neurons), 1)
        inter_neurons = n_neurons - command_neurons
        ```

    Combines a Liquid Time-Constant (LTC) cell with Ordinary Neural Circuits (ONCs). Paper references:

    - [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898)
    - [Reinforcement Learning with Ordinary Neural Circuits](https://proceedings.mlr.press/v119/hasani20a.html)
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        sparsity_level: float = 0.5,
        init_type: str | WeightInitType = "kaiming_uniform",
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            in_features (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes)
            out_features (int): number of out features (motor nodes)
            sparsity_level (float, optional): controls the connection sparsity
                between neurons.

                Must be a value between `[0.1, 0.9]` -

                - When `0.1` neurons are very dense.
                - When `0.9` they are very sparse.

            init_type (str, optional): the type of weight initialization
            device (torch.device, optional): the device to load tensors on
        """
        super().__init__()

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features
        self.device = device

        self.n_units = n_neurons + out_features  # inter + command + motor

        self.wiring = Wiring(
            in_features,
            n_neurons,
            out_features,
            sparsity_level=sparsity_level,
        )
        self.masks, self.counts = self.wiring.data()

        self.inter = SparseLinear(
            in_features,
            self.counts.inter,
            torch.abs(self.masks.inter.T),
            init_type=init_type,
            device=device,
        ).to(device)

        self.command = NCPLiquidCell(
            self.counts.inter,
            self.counts.command,
            self.masks.command,
            init_type=init_type,
            device=device,
        ).to(device)
        self.hidden_size = self.counts.command

        self.motor = SparseLinear(
            self.counts.command,
            self.counts.motor,
            torch.abs(self.masks.motor.T),
            init_type=init_type,
            device=device,
        ).to(device)

        self.act = nn.Mish()

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns:
            count (int): the total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns:
            count (int): the active parameter count.
        """
        return self._active_params

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
                f"Unsupported dimensionality: '{x.shape}'. Should be 2 dimensional with: '(batch_size, features)'."
            )

        x = x.to(dtype=torch.float32, device=self.device)

        batch_size, features = x.size()

        if h_state is None:
            h_state = torch.zeros(
                (batch_size, self.hidden_size),
                device=self.device,
            )

        # Batch -> (batch_size, out_features)
        x = self.act(self.inter(x))
        x, h_state = self.command(x, h_state.to(self.device))
        y_pred: torch.Tensor = self.motor(self.act(x))

        # Single item -> (out_features)
        if y_pred.shape[0] == 1:
            y_pred = y_pred.squeeze(0)

        # h_state -> (batch_size, n_units)
        return y_pred, h_state


class NCPNetwork(nn.Module):
    """
    A Neural Circuit Policy (NCP) Network with three layers:

    1. Inter (input) - a `SparseLinear` layer
    2. Command (hidden) - a `SparseLinear` layer
    3. Motor (output) - a `SparseLinear` layer

    Uses the Mish activation function between each layer.

    ??? note "Decision nodes"

        `inter` and `command` neurons are automatically calculated using:

        ```python
        command_neurons = max(int(0.4 * n_neurons), 1)
        inter_neurons = n_neurons - command_neurons
        ```

    Uses an Ordinary Neural Circuit (ONC) architecture without Liquid dynamics.
    Paper references:

    - [Reinforcement Learning with Ordinary Neural Circuits](https://proceedings.mlr.press/v119/hasani20a.html)
    - [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681)
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        sparsity_level: float = 0.5,
        init_type: str | WeightInitType = "kaiming_uniform",
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            in_features (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes)
            out_features (int): number of out features (motor nodes)
            sparsity_level (float, optional): controls the connection sparsity
                between neurons.

                Must be a value between `[0.1, 0.9]` -

                - When `0.1` neurons are very dense.
                - When `0.9` they are very sparse.

            init_type (str, optional): the type of weight initialization
            device (torch.device, optional): the device to load tensors on
        """
        super().__init__()

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features
        self.device = device

        self.n_units = n_neurons + out_features  # inter + command + motor

        self.wiring = Wiring(
            in_features,
            n_neurons,
            out_features,
            sparsity_level=sparsity_level,
        )
        self.masks, self.counts = self.wiring.data()

        self.ncp = nn.Sequential(
            SparseLinear(
                in_features,
                self.counts.inter,
                torch.abs(self.masks.inter.T),
                init_type=init_type,
                device=device,
            ),
            nn.Mish(),
            SparseLinear(
                self.counts.inter,
                self.counts.command,
                torch.abs(self.masks.command.T),
                init_type=init_type,
                device=device,
            ),
            nn.Mish(),
            SparseLinear(
                self.counts.command,
                self.counts.motor,
                torch.abs(self.masks.motor.T),
                init_type=init_type,
                device=device,
            ),
        ).to(device)

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns:
            count (int): the total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns:
            count (int): the active parameter count.
        """
        return self._active_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Parameters:
            x (torch.Tensor): an input tensor of shape: `(batch_size, features)`.

                - `batch_size` the number of samples per timestep.
                - `features` the features at each timestep (e.g.,
                image features, joint coordinates, word embeddings, raw amplitude
                values).

        Returns:
            y_pred (torch.Tensor): the network prediction. When `batch_size=1`. Out shape is `(out_features)`. Otherwise, `(batch_size, out_features)`.
        """
        if x.dim() != 2:
            raise ValueError(
                f"Unsupported dimensionality: '{x.shape}'. Should be 2 dimensional with: '(batch_size, features)'."
            )

        x = x.to(dtype=torch.float32, device=self.device)

        # Batch -> (batch_size, out_features)
        y_pred: torch.Tensor = self.ncp(x)

        # Single item -> (out_features)
        if y_pred.shape[0] == 1:
            y_pred = y_pred.squeeze(0)

        return y_pred
