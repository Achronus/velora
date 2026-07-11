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

from velora.nn.sparse import SparseLinear


class NCPLiquidCell(nn.Module):
    """
    A Liquid Time-Constant (LTC) cell using a Closed-form (CfC) approach.

    The LTC cell follows the closed-form continuous-depth
    (CFC; Equation 10) solution from the paper:
    [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898).

    Equation:
    $$
    x(t) =
        \\sigma(-f(x, I, θ_f), t) \\; g(x, I, θ_g)
        + \\left[ 1 - \\sigma(-[\\;f(x, I, θ_f)\\;]\\;t) \\right] \\; h(x, I, θ_h)
    $$

    Parameters
    ----------
    in_features : int
        Number of input nodes
    n_hidden : int
        Number of hidden nodes
    mask : torch.Tensor
        A matrix of sparse connections usually containing a combination
        of `[-1, 1, 0]` values
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: torch.Tensor,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.n_hidden = n_hidden
        self.head_size = n_hidden + in_features

        self.tanh = nn.Tanh()  # Bounded: [-1, 1]
        self.sigmoid = nn.Sigmoid()  # Bounded: [0, 1]

        mask = self._prep_mask(mask)

        self.g_head = SparseLinear(self.head_size, self.n_hidden, mask)
        self.h_head = SparseLinear(self.head_size, self.n_hidden, mask)

        # LTC heads (f)
        self.f_head_to_g = SparseLinear(self.head_size, self.n_hidden, mask)
        self.f_head_to_h = SparseLinear(self.head_size, self.n_hidden, mask)

    def _prep_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Utility method that preprocesses mask to match layer size.

        Adds hidden-to-hidden recurrent connections for continuous-time
        dynamics.

        Note -
            Performs two operations:

            1. Adds a padded matrix of 1s as extra columns to the mask in shape
               `(n_hidden, n_hidden)` for dense recurrent connections
            2. Gets the absolute values of the mask to maintain weight stability,
               converting `-1` to `1`

        Parameters
        ----------
        mask : torch.Tensor
            Weight sparsity mask of shape `(n_hidden, in_features)`

        Returns
        -------
        mask : torch.Tensor
            An updated mask of shape `(n_hidden, head_size)`
        """
        extra_nodes = torch.ones((self.n_hidden, self.n_hidden))
        mask = torch.cat([mask, extra_nodes], dim=1)
        return torch.abs(mask)

    def _timescale(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Overridable method. Modulates the timespan before temporal gating.

        Returns `ts` unchanged. Subclasses can override this to apply
        custom timescale dynamics (e.g., per-channel decay rates).

        Parameters
        ----------
        x : torch.Tensor
            Combined input and hidden state values of shape `(B, head_size)`
        ts : torch.Tensor
            Time elapsed since previous timestep

        Returns
        -------
        ts : torch.Tensor
            The modulated timespan
        """
        return ts

    def _new_hidden(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Helper method that computes the new hidden state.

        Parameters
        ----------
        x : torch.Tensor
            Combined input and hidden state values of shape `(B, head_size)`
        ts : torch.Tensor
            Time elapsed since previous timestep

        Returns
        -------
        hidden : torch.Tensor
            A new hidden state of shape `(B, n_hidden)`
        """
        g_head = self.tanh(self.g_head(x))  # g(x, I, θ_g)
        h_head = self.tanh(self.h_head(x))  # h(x, I, θ_h)

        fh_g = self.f_head_to_g(x)
        fh_h = self.f_head_to_h(x)

        # [1 - σ(-[f(x, I, θf)], t)]
        gate_out = self.sigmoid(fh_g * self._timescale(x, ts) + fh_h)
        f_head = 1.0 - gate_out  # σ(-f(x, I, θf), t)

        return g_head * f_head + gate_out * h_head

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        ts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the cell.

        Uses `ts` to control the temporal gating mechanism for
        continuous-time dynamics between hidden states.

        Parameters
        ----------
        x : torch.Tensor
            Input values of shape `(B, in_features)`
        hidden : torch.Tensor
            Current hidden state of shape `(B, n_hidden)`
        ts : torch.Tensor
            Time elapsed since previous timestep. A scalar, or a
            per-sample tensor broadcastable to `(B, n_hidden)`

        Returns
        -------
        y_pred : torch.Tensor
            The cell prediction of shape `(B, n_hidden)`
        h_state : torch.Tensor
            The new hidden state of shape `(B, n_hidden)`
        """
        x = torch.cat([x, hidden], dim=1)

        new_hidden = self._new_hidden(x, ts)
        return new_hidden, new_hidden


class DecayLiquidCell(NCPLiquidCell):
    """
    An expanded CfC-LTC with per-channel independent decay rates that act as
    a lightweight attention buffer.

    Each neuron maintains its own temporal sensitivity α_i(x)
    that modules its own timescale for reactive control and strategic memory.

    Neurons with α near:
      - `0` ignore the timespan (fast decay, reactive)
      - `1` fully incorporate it (slow decay, memory)

    Acts as an adaptation of [Kimi Delta Attention's (KDAs)](https://arxiv.org/abs/2510.26692) `Diag(α_t)` fine-grained gating for the CfC continuous-time setting.

    Equation:
    $$
        \\alpha(x, I) = \\sigma\\!\\left( W_{\\alpha}^{\\uparrow} \\;
        \\tanh\\!\\left( W_{\\alpha}^{\\downarrow} [x, I] \\right) \\right)
    $$

    $$
    x(t) =
        \\sigma(-f(x, I, θ_f), \\; t \\cdot \\alpha) \\;
        g(x, I, θ_g) + \\left[ 1 - \\sigma(-[\\;f(x, I, θ_f)\\;] \\;
        t \\cdot \\alpha) \\right] \\;
        h(x, I, θ_h)
    $$

    Parameters
    ----------
    in_features : int
        Number of input nodes
    n_hidden : int
        Number of hidden nodes
    mask : torch.Tensor
        A matrix of sparse connections usually containing a combination
        of `[-1, 1, 0]` values
    alpha_rank : int (optional)
        Rank of the low-rank α projection. Default is `min(n_hidden, 4)`
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: torch.Tensor,
        *,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__(in_features, n_hidden, mask)

        self.alpha_rank = min(n_hidden, 4) if alpha_rank is None else alpha_rank

        # Per-channel decay: low-rank projection (head_size → rank → n_hidden)
        self.alpha_down = nn.Linear(self.head_size, self.alpha_rank)
        self.alpha_up = nn.Linear(self.alpha_rank, self.n_hidden)

    def _timescale(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Modulates the timespan with per-channel decay rates `α` in `[0, 1]`.

        Parameters
        ----------
        x : torch.Tensor
            Combined input and hidden state values of shape `(B, head_size)`
        ts : torch.Tensor
            Time elapsed since previous timestep

        Returns
        -------
        ts : torch.Tensor
            The decayed timespan of shape `(B, n_hidden)`
        """
        alpha = self.sigmoid(self.alpha_up(self.tanh(self.alpha_down(x))))
        return ts * alpha


class DeltaErasureLiquidCell(NCPLiquidCell):
    """
    An expanded CfC-LTC with delta-rule selective memory erasure.

    Computes what the current input "expects" the hidden state to be
    and partially corrects each hidden dimension toward that target.
    This enables the cell to actively revise stale beliefs when it
    encounters a surprising state transition.

    Inspired by [Kimi Delta Attention's (KDAs)](https://arxiv.org/abs/2510.26692)
    `(I - β_t k_t k_t^T)` erasure term.

    Equation:
    $$
        \\hat{I} = I + \\sigma(\\beta(x, I, θ_{\\beta}))
        \\cdot \\left( \\tanh(r(x, I, θ_r)) - I \\right)
    $$

    $$
    x(t) =
        \\sigma(-f(x, \\hat{I}, θ_f), t) \\; g(x, \\hat{I}, θ_g)
        + \\left[ 1 - \\sigma(-[\\;f(x, \\hat{I}, θ_f)\\;]\\;t) \\right] \\; h(x, \\hat{I}, θ_h)
    $$

    Parameters
    ----------
    in_features : int
        Number of input nodes
    n_hidden : int
        Number of hidden nodes
    mask : torch.Tensor
        A matrix of sparse connections usually containing a combination
        of `[-1, 1, 0]` values
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: torch.Tensor,
    ) -> None:
        super().__init__(in_features, n_hidden, mask)

        mask = self._prep_mask(mask)

        # Delta-rule erasure heads
        self.reconstruct_head = SparseLinear(self.head_size, self.n_hidden, mask)
        self.beta_head = SparseLinear(self.head_size, self.n_hidden, mask)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        ts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the cell.

        Corrects the hidden state toward the input's expectation before
        applying the CfC temporal gating.

        Parameters
        ----------
        x : torch.Tensor
            Input values of shape `(B, in_features)`
        hidden : torch.Tensor
            Current hidden state of shape `(B, n_hidden)`
        ts : torch.Tensor
            Time elapsed since previous timestep. A scalar, or a
            per-sample tensor broadcastable to `(B, n_hidden)`

        Returns
        -------
        y_pred : torch.Tensor
            The cell prediction of shape `(B, n_hidden)`
        h_state : torch.Tensor
            The new hidden state of shape `(B, n_hidden)`
        """
        x_cat = torch.cat([x, hidden], dim=1)

        # Get hidden expectation
        expected = self.tanh(self.reconstruct_head(x_cat))

        # Per-dimension correction strength
        beta = self.sigmoid(self.beta_head(x_cat))

        # Selective correction: β=0 keeps old, β=1 overwrites
        hidden_corrected = hidden + beta * (expected - hidden)

        # CfC with corrected hidden
        x = torch.cat([x, hidden_corrected], dim=1)

        new_hidden = self._new_hidden(x, ts)
        return new_hidden, new_hidden


class AdaptiveLiquidCell(DecayLiquidCell, DeltaErasureLiquidCell):
    """
    An enhanced CfC-LTC that combines per-channel decay AND delta-rule
    erasure.

    Provides the cell with two complementary abilities:
        - Erasure - hidden state belief correct (WHAT the cell remembers)
        - Per-channel decay - per-neuron attention timescales
          (HOW LONG the cell remembers information)

    $$
    \\hat{I} = I + \\sigma(\\beta(x, I, θ_{\\beta}))
        \\cdot \\left( \\tanh(r(x, I, θ_r)) - I \\right)
    $$

    $$
    \\alpha(x, \\hat{I}) = \\sigma\\!\\left( W_{\\alpha}^{\\uparrow} \\; \\tanh\\!\\left( W_{\\alpha}^{\\downarrow} [x, \\hat{I}] \\right) \\right)
    $$

    $$
    x(t) =
        \\sigma(-f(x, \\hat{I}, θ_f), \\; t \\cdot \\alpha) \\; g(x, \\hat{I}, θ_g)
        + \\left[ 1 - \\sigma(-[\\;f(x, \\hat{I}, θ_f)\\;] \\; t \\cdot \\alpha) \\right] \\; h(x, \\hat{I}, θ_h)
    $$

    Parameters
    ----------
    in_features : int
        Number of input nodes
    n_hidden : int
        Number of hidden nodes
    mask : torch.Tensor
        A matrix of sparse connections usually containing a combination
        of `[-1, 1, 0]` values
    alpha_rank : int (optional)
        Rank of the low-rank α projection. Default is `min(n_hidden, 4)`
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: torch.Tensor,
        *,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__(in_features, n_hidden, mask, alpha_rank=alpha_rank)
