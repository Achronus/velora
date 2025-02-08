import torch
import torch.nn as nn
import torch.nn.functional as F


class NCPLiquidCell(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: torch.Tensor,
    ) -> None:
        """
        A Neural Circuit Policy (NCP) Liquid Time-Constant (LTC) cell.

        The LTC cell follows the closed-form continuous-depth
        (CFC; Equation 10) solution from the paper:
        [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898).

        Plus, it follows an Ordinary Neural Circuit (ONC) approach from this paper:
        [Reinforcement Learning with Ordinary Neural Circuits](https://proceedings.mlr.press/v119/hasani20a.html).

        Same as `LTCCell` but with a weight sparsity mask instead of a
        backbone. Intended for small-scale architectures.

        Equation:
        x(t) =
            σ(-f(x, I, θ_f), t) g(x, I, θ_g)
            + [1 - σ(-[f(x, I, θ_f)]t)] h(x, I, θ_h)

        Parameters:
            in_features (int): number of input nodes
            n_hidden (int): number of hidden nodes
            mask (torch.Tensor): a matrix of sparse connections
                usually containing a combination of `[-1, 1, 0]`
        """

        super().__init__()

        self.in_features = in_features
        self.n_hidden = n_hidden
        self.head_size = n_hidden + in_features

        # Absolute to maintain masking (-1 -> 1)
        self.sparsity_mask = nn.Parameter(
            self._prep_mask(mask),
            requires_grad=False,
        )

        self.tanh = nn.Tanh()  # Bounded: [-1, 1]
        self.sigmoid = nn.Sigmoid()  # Bounded: [0, 1]

        self.g_head = nn.Linear(self.head_size, n_hidden)
        self.h_head = nn.Linear(self.head_size, n_hidden)

        # LTC heads (f)
        self.f_head_to_g = nn.Linear(self.head_size, n_hidden)
        self.f_head_to_h = nn.Linear(self.head_size, n_hidden)

    def _prep_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses mask to match head size.

        Performs three operations:
        1. Adds a padded matrix of 1s to end of mask in shape
            `(n_extras, n_extras)` where `n_extras=mask.shape[1]`
        2. Transposes mask from col matrix -> row matrix
        3. Gets the absolute values of the mask (swapping -1 -> 1)
        """
        n_extras = mask.shape[1]
        extra_nodes = torch.ones((n_extras, n_extras))
        mask = torch.concatenate([mask, extra_nodes])
        return torch.abs(mask.T)

    def _sparse_head(self, x: torch.Tensor, head: nn.Linear) -> torch.Tensor:
        """Computes the output for a sparsity mask head."""
        return F.linear(x, head.weight * self.sparsity_mask, head.bias)

    def _new_hidden(
        self,
        x: torch.Tensor,
        g_out: torch.Tensor,
        h_out: torch.Tensor,
        ts: torch.Tensor | float,
    ) -> torch.Tensor:
        """
        Computes the new hidden state.

        Parameters:
            x (torch.Tensor): input values
            g_out (torch.Tensor): g_head output
            h_out (torch.Tensor): h_head output
            ts (torch.Tensor | float): current timespan between events

        Returns:
            hidden (torch.Tensor): a new hidden state
        """
        g_head = self.tanh(g_out)  # g(x, I, θ_g)
        h_head = self.tanh(h_out)  # h(x, I, θ_h)

        fh_g = self.f_head_to_g(x)
        fh_h = self.f_head_to_h(x)

        gate_out = self.sigmoid(fh_g * ts + fh_h)  # [1 - σ(-[f(x, I, θf)], t)]
        f_head = 1.0 - gate_out  # σ(-f(x, I, θf), t)

        return g_head * f_head + gate_out * h_head

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, ts: torch.Tensor | float
    ) -> torch.Tensor:
        """
        Performs a forward pass through the cell.

        Parameters:
            x (torch.Tensor): input values
            hidden (torch.Tensor): current hidden state
            ts (torch.Tensor | float): current timespan between events

        Returns:
            hidden (torch.Tensor): a new hidden state
        """
        x = torch.cat([x, hidden], 1)

        g_out = self._sparse_head(x, self.g_head)
        h_out = self._sparse_head(x, self.h_head)

        return self._new_hidden(x, g_out, h_out, ts)
