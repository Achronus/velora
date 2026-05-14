# Copyright 2025 Achronus
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

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from velora.lnn.constants import DEFAULT_HIDDEN_INIT
from velora.lnn.sparse import SparseLinear


class CellConfig:
    """
    Configuration for creating `NCPLiquidCell` variants.

    Captures the cell class and any variant-specific keyword arguments.
    Used by `BaseCfC` to construct cells at each layer.

    Should not be instantiated directly — use `NCPLiquidCell.config()`
    or a subclass's `.config()` class method instead.

    Parameters
    ----------
    cell_type : type[NCPLiquidCell]
        The cell class to use
    **kwargs : Any
        Variant-specific keyword arguments passed to the cell constructor.
        E.g., `alpha_rank=8` for `AdaptiveLiquidCell`
    """

    def __init__(
        self,
        cell_type: type["NCPLiquidCell"],
        **kwargs: Any,
    ) -> None:
        self.cell_type = cell_type
        self.kwargs = kwargs

    def build(
        self,
        in_features: int,
        n_hidden: int,
        mask: jax.Array,
        *,
        rngs: nnx.Rngs,
        init_type: Initializer,
    ) -> "NCPLiquidCell":
        """
        Construct a cell instance with the stored configuration.

        Parameters
        ----------
        in_features : int
            Number of input nodes
        n_hidden : int
            Number of hidden nodes
        mask : jax.Array
            A matrix of sparse connections
        rngs : flax.nnx.Rngs
            Random number generator key
        init_type : flax.nnx.nn.initializers
            Initializer function for the weight matrix

        Returns
        -------
        cell : NCPLiquidCell
            A configured cell instance
        """
        return self.cell_type(
            in_features,
            n_hidden,
            mask,
            rngs=rngs,
            init_type=init_type,
            **self.kwargs,
        )


class NCPLiquidCell(nnx.Module):
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
    mask : jax.Array
        A matrix of sparse connections usually containing a combination
        of `[-1, 1, 0]` values
    rngs : flax.nnx.Rngs (optional)
        Random number generator key.
        Must have a `params=[value]` attribute
    init_type : flax.nnx.nn.initializers (optional)
        Initializer function for the weight matrix.
        Default is `lecun_uniform()`
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: jax.Array,
        *,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
        init_type: Initializer = DEFAULT_HIDDEN_INIT,
    ) -> None:
        self.in_features = in_features
        self.n_hidden = n_hidden
        self.head_size = n_hidden + in_features
        self.init_type = init_type
        self.rngs = rngs

        # Absolute to maintain masking (-1 -> 1)
        self.sparsity_mask = self._prep_mask(mask)

        self.tanh = nnx.tanh  # Bounded: [-1, 1]
        self.sigmoid = nnx.sigmoid  # Bounded: [0, 1]

        self.g_head = self._make_layer()
        self.h_head = self._make_layer()

        # LTC heads (f)
        self.f_head_to_g = self._make_layer()
        self.f_head_to_h = self._make_layer()

    @classmethod
    def config(cls) -> CellConfig:
        """
        Returns a `CellConfig` for this cell type.

        Returns
        -------
        config : CellConfig
            A cell configuration for constructing this cell variant
        """
        return CellConfig(cls)

    def _make_layer(self) -> SparseLinear:
        """
        Helper method that creates a new `SparseLinear` layer.

        The layer is configured with the following values:

        - `in_features` - `self.n_hidden + self.in_features`
        - `out_features` - `self.n_hidden`
        - `mask` - `self.sparsity_mask`

        Returns
        -------
        layer : SparseLinear
            A `SparseLinear` layer
        """
        return SparseLinear(
            self.head_size,
            self.n_hidden,
            self.sparsity_mask,
            rngs=self.rngs,
            hidden_init=self.init_type,
        )

    def _prep_mask(self, mask: jax.Array) -> jax.Array:
        """
        Utility method that preprocesses mask to match layer size.

        Adds hidden-to-hidden recurrent connections for continuous-time
        dynamics.

        Note -
            Performs two operations:

            1. Adds a padded matrix of 1s to end of mask in shape
               `(n_hidden, n_hidden)` for dense recurrent connections
            2. Gets the absolute values of the mask to maintain weight stability,
               converting `-1` to `1`

        Parameters
        ----------
        mask : jax.Array
            Weight sparsity mask

        Returns
        -------
        mask : jax.Array
            An updated mask
        """
        extra_nodes = jnp.ones((self.n_hidden, self.n_hidden))
        mask = jnp.concat([mask, extra_nodes])
        return jnp.abs(mask)

    def _new_hidden(
        self,
        x: jax.Array,
        g_out: jax.Array,
        h_out: jax.Array,
        ts: jax.Array,
    ) -> jax.Array:
        """
        Helper method that computes the new hidden state.

        Parameters
        ----------
        x : jax.Array
            Input values
        g_out : jax.Array
            g_head output
        h_out : jax.Array
            h_head output
        ts : jax.Array
            Time elapsed since previous timestep

        Returns
        -------
        hidden : jax.Array
            A new hidden state
        """
        g_head = self.tanh(g_out)  # g(x, I, θ_g)
        h_head = self.tanh(h_out)  # h(x, I, θ_h)

        fh_g = self.f_head_to_g(x)
        fh_h = self.f_head_to_h(x)

        gate_out = self.sigmoid(fh_g * ts + fh_h)  # [1 - σ(-[f(x, I, θf)], t)]
        f_head = 1.0 - gate_out  # σ(-f(x, I, θf), t)

        return g_head * f_head + gate_out * h_head

    def __call__(
        self, x: jax.Array, hidden: jax.Array, timespans: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Performs a forward pass through the cell.

        Uses `timespans` to control the temporal gating mechanism for
        continuous-time dynamics between hidden states.

        Parameters
        ----------
        x : jax.Array
            Input values
        hidden : jax.Array
            Current hidden state
        timespans : jax.Array
            Time elapsed since previous timestep.
            Shape should be `(T,)`

        Returns
        -------
        y_pred : jax.Array
            The cell prediction
        h_state : jax.Array
            The hidden state
        """
        x = jnp.concat([x, hidden], axis=1)

        g_out = self.g_head(x)
        h_out = self.h_head(x)

        new_hidden = self._new_hidden(x, g_out, h_out, timespans)
        return new_hidden, new_hidden

    def diagnostics(
        self, x: jax.Array, hidden: jax.Array, name: str = "cell"
    ) -> Dict[str, float]:
        """
        Extract mechanism-specific metrics for logging.

        Parameters
        ----------
        x : jax.Array
            Current input `(B, in_features)`
        hidden : jax.Array
            Current hidden state `(B, n_hidden)`
        name : str (optional)
            Layer/cell name for logging

        Returns
        -------
        metrics : Dict[str, float]
            Diagnostic scalars for logging
        """
        x_cat = jnp.concat([x, hidden], axis=1)

        fh_g = self.f_head_to_g(x_cat)
        fh_h = self.f_head_to_h(x_cat)

        # The uniform gate
        gate = self.sigmoid(fh_g + fh_h)  # ts=1.0

        return {
            f"{name}/gate_mean": float(jnp.mean(gate)),
            f"{name}/gate_std": float(jnp.std(gate)),
            f"{name}/gate_min": float(jnp.min(gate)),
            f"{name}/gate_max": float(jnp.max(gate)),
        }


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
    mask : jax.Array
        A matrix of sparse connections usually containing a combination
        of `[-1, 1, 0]` values
    rngs : flax.nnx.Rngs (optional)
        Random number generator key.
        Must have a `params=[value]` attribute
    init_type : flax.nnx.nn.initializers (optional)
        Initializer function for the weight matrix.
        Default is `lecun_uniform()`
    alpha_rank : int (optional)
        Rank of the low-rank α projection. Default is `min(n_hidden, 4)`
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: jax.Array,
        *,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
        init_type: Initializer = DEFAULT_HIDDEN_INIT,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__(
            in_features,
            n_hidden,
            mask,
            rngs=rngs,
            init_type=init_type,
        )

        self.alpha_rank = alpha_rank or min(n_hidden, 4)

        # Per-channel decay: low-rank projection (head_size → rank → n_hidden)
        self.alpha_down = nnx.Linear(self.head_size, self.alpha_rank, rngs=rngs)
        self.alpha_up = nnx.Linear(self.alpha_rank, self.n_hidden, rngs=rngs)

    @classmethod
    def config(cls, *, alpha_rank: int | None = None) -> CellConfig:
        """
        Returns a `CellConfig` for this cell type.

        Parameters
        ----------
        alpha_rank : int (optional)
            Rank of the low-rank α projection.
            Default is `None` (uses `min(n_hidden, 4)`)

        Returns
        -------
        config : CellConfig
            A cell configuration for constructing this cell variant
        """
        return CellConfig(cls, alpha_rank=alpha_rank)

    def _new_hidden(
        self,
        x: jax.Array,
        g_out: jax.Array,
        h_out: jax.Array,
        ts: jax.Array,
    ) -> jax.Array:
        g_head = self.tanh(g_out)  # g(x, I, θ_g)
        h_head = self.tanh(h_out)  # h(x, I, θ_h)

        fh_g = self.f_head_to_g(x)
        fh_h = self.f_head_to_h(x)

        # Per-channel decay rate in [0, 1]
        alpha = self.sigmoid(self.alpha_up(self.tanh(self.alpha_down(x))))  # type: ignore

        # Modulate timespan per-channel before temporal gating
        gate_out = self.sigmoid(fh_g * (ts * alpha) + fh_h)
        f_head = 1.0 - gate_out

        return g_head * f_head + gate_out * h_head

    def diagnostics(
        self, x: jax.Array, hidden: jax.Array, name: str = "cell"
    ) -> Dict[str, float]:
        x_cat = jnp.concat([x, hidden], axis=1)
        alpha = self.sigmoid(self.alpha_up(self.tanh(self.alpha_down(x_cat))))

        return {
            f"{name}/alpha_mean": float(jnp.mean(alpha)),
            f"{name}/alpha_std": float(jnp.std(alpha)),
            f"{name}/alpha_min": float(jnp.min(alpha)),
            f"{name}/alpha_max": float(jnp.max(alpha)),
        }


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
    mask : jax.Array
        A matrix of sparse connections usually containing a combination
        of `[-1, 1, 0]` values
    rngs : flax.nnx.Rngs (optional)
        Random number generator key.
        Must have a `params=[value]` attribute
    init_type : flax.nnx.nn.initializers (optional)
        Initializer function for the weight matrix.
        Default is `lecun_uniform()`
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: jax.Array,
        *,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
        init_type: Initializer = DEFAULT_HIDDEN_INIT,
    ) -> None:
        super().__init__(
            in_features,
            n_hidden,
            mask,
            rngs=rngs,
            init_type=init_type,
        )

        # Delta-rule erasure heads
        self.reconstruct_head = self._make_layer()
        self.beta_head = self._make_layer()

    def __call__(
        self, x: jax.Array, hidden: jax.Array, timespans: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        x_cat = jnp.concat([x, hidden], axis=1)

        # Get hidden expectation
        expected = self.tanh(self.reconstruct_head(x_cat))

        # Per-dimension correction strength
        beta = self.sigmoid(self.beta_head(x_cat))

        # Selective correction: β=0 keeps old, β=1 overwrites
        hidden_corrected = hidden + beta * (expected - hidden)

        # CfC with corrected hidden
        x = jnp.concat([x, hidden_corrected], axis=1)

        g_out = self.g_head(x)
        h_out = self.h_head(x)

        new_hidden = self._new_hidden(x, g_out, h_out, timespans)
        return new_hidden, new_hidden

    def diagnostics(
        self, x: jax.Array, hidden: jax.Array, name: str = "cell"
    ) -> Dict[str, float]:
        x_cat = jnp.concat([x, hidden], axis=1)

        expected = self.tanh(self.reconstruct_head(x_cat))
        beta = self.sigmoid(self.beta_head(x_cat))
        recon_error = jnp.mean(jnp.abs(expected - hidden))

        return {
            f"{name}/beta_mean": float(jnp.mean(beta)),
            f"{name}/beta_std": float(jnp.std(beta)),
            f"{name}/reconstruction_error": float(recon_error),
        }


class AdaptiveLiquidCell(NCPLiquidCell):
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
    mask : jax.Array
        A matrix of sparse connections usually containing a combination
        of `[-1, 1, 0]` values
    rngs : flax.nnx.Rngs (optional)
        Random number generator key.
        Must have a `params=[value]` attribute
    init_type : flax.nnx.nn.initializers (optional)
        Initializer function for the weight matrix.
        Default is `lecun_uniform()`
    alpha_rank : int (optional)
        Rank of the low-rank α projection. Default is `min(n_hidden, 4)`
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: jax.Array,
        *,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
        init_type: Initializer = DEFAULT_HIDDEN_INIT,
        alpha_rank: int | None = None,
    ) -> None:
        super().__init__(
            in_features,
            n_hidden,
            mask,
            rngs=rngs,
            init_type=init_type,
        )

        self.alpha_rank = alpha_rank or min(n_hidden, 4)

        # Per-channel decay: low-rank projection (head_size → rank → n_hidden)
        self.alpha_down = nnx.Linear(self.head_size, self.alpha_rank, rngs=rngs)
        self.alpha_up = nnx.Linear(self.alpha_rank, self.n_hidden, rngs=rngs)

        # Delta-rule erasure heads
        self.reconstruct_head = self._make_layer()
        self.beta_head = self._make_layer()

    @classmethod
    def config(cls, *, alpha_rank: int | None = None) -> CellConfig:
        """
        Returns a `CellConfig` for this cell type.

        Parameters
        ----------
        alpha_rank : int (optional)
            Rank of the low-rank α projection.
            Default is `None` (uses `min(n_hidden, 4)`)

        Returns
        -------
        config : CellConfig
            A cell configuration for constructing this cell variant
        """
        return CellConfig(cls, alpha_rank=alpha_rank)

    def _new_hidden(
        self,
        x: jax.Array,
        g_out: jax.Array,
        h_out: jax.Array,
        ts: jax.Array,
    ) -> jax.Array:
        g_head = self.tanh(g_out)  # g(x, I, θ_g)
        h_head = self.tanh(h_out)  # h(x, I, θ_h)

        fh_g = self.f_head_to_g(x)
        fh_h = self.f_head_to_h(x)

        # Per-channel decay rate in [0, 1]
        alpha = self.sigmoid(self.alpha_up(self.tanh(self.alpha_down(x))))  # type: ignore

        # Modulate timespan per-channel before temporal gating
        gate_out = self.sigmoid(fh_g * (ts * alpha) + fh_h)
        f_head = 1.0 - gate_out

        return g_head * f_head + gate_out * h_head

    def __call__(
        self, x: jax.Array, hidden: jax.Array, timespans: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        x_cat = jnp.concat([x, hidden], axis=1)

        # Get hidden expectation
        expected = self.tanh(self.reconstruct_head(x_cat))

        # Per-dimension correction strength
        beta = self.sigmoid(self.beta_head(x_cat))

        # Selective correction: β=0 keeps old, β=1 overwrites
        hidden_corrected = hidden + beta * (expected - hidden)

        # CfC with corrected hidden
        x = jnp.concat([x, hidden_corrected], axis=1)

        g_out = self.g_head(x)
        h_out = self.h_head(x)

        new_hidden = self._new_hidden(x, g_out, h_out, timespans)
        return new_hidden, new_hidden

    def diagnostics(
        self, x: jax.Array, hidden: jax.Array, name: str = "cell"
    ) -> Dict[str, float]:
        x_cat = jnp.concat([x, hidden], axis=1)

        # Erasure metrics
        expected = self.tanh(self.reconstruct_head(x_cat))
        beta = self.sigmoid(self.beta_head(x_cat))
        recon_error = jnp.mean(jnp.abs(expected - hidden))

        # Alpha distribution
        hidden_corrected = hidden + beta * (expected - hidden)
        x_corrected = jnp.concat([x, hidden_corrected], axis=1)
        alpha = self.sigmoid(self.alpha_up(self.tanh(self.alpha_down(x_corrected))))

        return {
            f"{name}/alpha_mean": float(jnp.mean(alpha).item()),
            f"{name}/alpha_std": float(jnp.std(alpha).item()),
            f"{name}/alpha_min": float(jnp.min(alpha).item()),
            f"{name}/alpha_max": float(jnp.max(alpha).item()),
            f"{name}/beta_mean": float(jnp.mean(beta).item()),
            f"{name}/beta_std": float(jnp.std(beta).item()),
            f"{name}/reconstruction_error": float(recon_error.item()),
        }
