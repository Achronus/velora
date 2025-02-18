from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCPLiquidCell(nn.Module):
    """
    A Neural Circuit Policy (NCP) Liquid Time-Constant (LTC) cell.

    The LTC cell follows the closed-form continuous-depth
    (CFC; Equation 10) solution from the paper:
    [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898).

    Plus, it follows an Ordinary Neural Circuit (ONC) approach from this paper:
    [Reinforcement Learning with Ordinary Neural Circuits](https://proceedings.mlr.press/v119/hasani20a.html).

    Equation:
    $$
    x(t) =
        \\sigma(-f(x, I, θ_f), t) \\; g(x, I, θ_g)
        + \\left[ 1 - \\sigma(-[\\;f(x, I, θ_f)\\;]\\;t) \\right] \\; h(x, I, θ_h)
    $$
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            in_features (int): number of input nodes.
            n_hidden (int): number of hidden nodes.
            mask (torch.Tensor): a matrix of sparse connections
                usually containing a combination of `[-1, 1, 0]`.
            device (torch.device, optional): the device to load tensors on.
        """

        super().__init__()

        self.in_features = in_features
        self.n_hidden = n_hidden
        self.head_size = n_hidden + in_features
        self.device = device

        # Absolute to maintain masking (-1 -> 1)
        self.sparsity_mask = nn.Parameter(
            self._prep_mask(mask.to(device)),
            requires_grad=False,
        )

        self.tanh = nn.Tanh()  # Bounded: [-1, 1]
        self.sigmoid = nn.Sigmoid()  # Bounded: [0, 1]

        self.g_head = nn.Linear(self.head_size, n_hidden, device=device)
        self.h_head = nn.Linear(self.head_size, n_hidden, device=device)

        # LTC heads (f)
        self.f_head_to_g = nn.Linear(self.head_size, n_hidden, device=device)
        self.f_head_to_h = nn.Linear(self.head_size, n_hidden, device=device)

        # Hidden state projection
        self.proj = nn.Linear(self.head_size, n_hidden, device=device)

    def _prep_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Utility method. Preprocesses mask to match head size.

        !!! note "Performs three operations"

            1. Adds a padded matrix of 1s to end of mask in shape
                `(n_extras, n_extras)` where `n_extras=mask.shape[1]`
            2. Transposes mask from col matrix -> row matrix
            3. Gets the absolute values of the mask (swapping `-1 -> 1`)

        Parameters:
            mask (torch.Tensor): weight sparsity mask.

        Returns:
            mask (torch.Tensor): an updated mask.
        """
        n_extras = mask.shape[1]
        extra_nodes = torch.ones((n_extras, n_extras), device=self.device)
        mask = torch.concatenate([mask, extra_nodes])
        return torch.abs(mask.T)

    def _sparse_head(self, x: torch.Tensor, head: nn.Linear) -> torch.Tensor:
        """
        Utility method. Computes the output for a sparsity mask head.

        Parameters:
            x (torch.Tensor): layer inputs.
            head (nn.Linear): linear head to use with sparse connections.

        Returns:
            y_pred (torch.Tensor): sparse head prediction.
        """
        return F.linear(
            x.to(torch.float32),
            head.weight * self.sparsity_mask,
            head.bias,
        )

    def _new_hidden(
        self, x: torch.Tensor, g_out: torch.Tensor, h_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Helper method. Computes the new hidden state.

        Parameters:
            x (torch.Tensor): input values.
            g_out (torch.Tensor): g_head output.
            h_out (torch.Tensor): h_head output.

        Returns:
            hidden (torch.Tensor): a new hidden state
        """
        g_head = self.tanh(g_out)  # g(x, I, θ_g)
        h_head = self.tanh(h_out)  # h(x, I, θ_h)

        fh_g = self._sparse_head(x, self.f_head_to_g)
        fh_h = self._sparse_head(x, self.f_head_to_h)

        gate_out = self.sigmoid(fh_g + fh_h)  # [1 - σ(-[f(x, I, θf)], t)]
        f_head = 1.0 - gate_out  # σ(-f(x, I, θf), t)

        return g_head * f_head + gate_out * h_head

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the cell.

        Parameters:
            x (torch.Tensor): input values.
            hidden (torch.Tensor): current hidden state.

        Returns:
            y_pred (torch.Tensor): the cell prediction.
            h_state (torch.Tensor): the hidden state.
        """
        x, hidden = x.to(self.device), hidden.to(self.device)
        x = torch.cat([x, hidden], dim=1)

        g_out = self._sparse_head(x, self.g_head)
        h_out = self._sparse_head(x, self.h_head)

        new_hidden = self._new_hidden(x, g_out, h_out)
        y_pred = self._sparse_head(x, self.proj) + new_hidden
        return y_pred, new_hidden
