from typing import Tuple

import chex
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from velora.constants import DEFAULT_HIDDEN_INIT
from velora.models.sparse import SparseLinear


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

    Parameters:
        in_features (int): number of input nodes.
        n_hidden (int): number of hidden nodes.
        mask (jax.Array): a matrix of sparse connections
            usually containing a combination of `[-1, 1, 0]` values.
        rngs (flax.nnx.Rngs, optional): random number generator key.
            Must have a `params=[value]` attribute
        init_type (flax.nnx.nn.initializers, optional): initializer function for the
            weight matrix. Default is `lecun_uniform()`
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        mask: chex.Array,
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

    def _make_layer(self) -> SparseLinear:
        """
        Helper method. Creates a new `SparseLinear` layer with the following values:

        - `in_features` - `self.n_hidden + self.in_features`.
        - `out_features` - `self.n_hidden`.
        - `mask` - `self.sparsity_mask`.

        Returns:
            layer (SparseLinear): a `SparseLinear` layer.
        """
        return SparseLinear(
            self.head_size,
            self.n_hidden,
            self.sparsity_mask,
            rngs=self.rngs,
            hidden_init=self.init_type,
        )

    def _prep_mask(self, mask: chex.Array) -> chex.Array:
        """
        Utility method. Preprocesses mask to match head size.

        !!! note "Performs two operations"

            1. Adds a padded matrix of 1s to end of mask in shape
                `(n_extras, n_extras)` where `n_extras=mask.shape[1]`
            3. Gets the absolute values of the mask (sanity check)

        Parameters:
            mask (jax.Array): weight sparsity mask.

        Returns:
            mask (jax.Array): an updated mask.
        """
        extra_nodes = jnp.ones((self.n_hidden, self.n_hidden))
        mask = jnp.concat([mask, extra_nodes])
        return jnp.abs(mask)

    def _new_hidden(
        self,
        x: chex.Array,
        g_out: chex.Array,
        h_out: chex.Array,
        ts: chex.Array,
    ) -> chex.Array:
        """
        Helper method. Computes the new hidden state.

        Parameters:
            x (jax.Array): input values.
            g_out (jax.Array): g_head output.
            h_out (jax.Array): h_head output.
            ts (jax.Array): time elapsed since previous timestep.

        Returns:
            hidden (jax.Array): a new hidden state
        """
        g_head = self.tanh(g_out)  # g(x, I, θ_g)
        h_head = self.tanh(h_out)  # h(x, I, θ_h)

        fh_g = self.f_head_to_g(x)
        fh_h = self.f_head_to_h(x)

        gate_out = self.sigmoid(fh_g * ts + fh_h)  # [1 - σ(-[f(x, I, θf)], t)]
        f_head = 1.0 - gate_out  # σ(-f(x, I, θf), t)

        return g_head * f_head + gate_out * h_head

    def __call__(
        self, x: chex.Array, hidden: chex.Array, timespans: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Performs a forward pass through the cell.

        Uses `timespans` to control the temporal gating mechanism for
        continuous-time dynamics between hidden states.

        Parameters:
            x (jax.Array): input values.
            hidden (jax.Array): current hidden state.
            timespans (jax.Array): time elapsed since previous timestep.
                Shape should be `(T,)`

        Returns:
            y_pred (jax.Array): the cell prediction.
            h_state (jax.Array): the hidden state.
        """
        x = jnp.concat([x, hidden], axis=1)

        g_out = self.g_head(x)
        h_out = self.h_head(x)

        new_hidden = self._new_hidden(x, g_out, h_out, timespans)
        return new_hidden, new_hidden
