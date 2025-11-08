from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from velora.constants import DEFAULT_HIDDEN_INIT
from velora.models.lnn.cell import NCPLiquidCell
from velora.models.sparse import SparseLinear
from velora.utils.nn import active_parameters, total_parameters
from velora.wiring import build_ncp_wiring


class LiquidNCPNetwork(nnx.Module):
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

    Parameters:
        in_features (int): number of inputs (sensory nodes)
        n_neurons (int): number of decision nodes (inter and command nodes)
        out_features (int): number of out features (motor nodes)
        seed (int, optional): random number generator seed. Default is `28`
        sparsity_level (float, optional): controls the connection sparsity
            between neurons.

            Must be a value between `[0.1, 0.9]` -

            - When `0.1` neurons are very dense.
            - When `0.9` they are very sparse.

        init_type (flax.nnx.nn.initializers, optional): initializer function for the
            weight matrix. Default is `lecun_uniform()`
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        seed: int = 28,
        sparsity_level: float = 0.5,
        init_type: Initializer = DEFAULT_HIDDEN_INIT,
    ) -> None:
        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features
        self.seed = seed
        self.rngs = nnx.Rngs(params=seed)

        self.n_units = n_neurons + out_features  # inter + command + motor

        self.wiring = nnx.data(
            build_ncp_wiring(
                in_features,
                n_neurons,
                out_features,
                seed=seed,
                sparsity_level=sparsity_level,
            )
        )

        self.inter = SparseLinear(
            in_features,
            self.wiring.inter.n_nodes,
            jnp.abs(self.wiring.inter.mask.T),
            rngs=self.rngs,
            hidden_init=init_type,
        )

        self.command = NCPLiquidCell(
            self.wiring.inter.n_nodes,
            self.wiring.command.n_nodes,
            self.wiring.command.mask,
            rngs=self.rngs,
            init_type=init_type,
        )
        self.hidden_size = self.wiring.command.n_nodes

        self.motor = SparseLinear(
            self.wiring.command.n_nodes,
            self.wiring.motor.n_nodes,
            jnp.abs(self.wiring.motor.mask.T),
            rngs=self.rngs,
            hidden_init=init_type,
        )

        self.act = jax.nn.mish

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

    def __call__(
        self, x: jax.Array, h_state: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Performs a forward pass through the network.

        Parameters:
            x (jax.Array): an input tensor of shape: `(batch_size, features)`.

                - `batch_size` the number of samples per timestep.
                - `features` the features at each timestep (e.g.,
                image features, joint coordinates, word embeddings, raw amplitude
                values).
            h_state (jax.Array, optional): initial hidden state of the RNN with
                shape: `(batch_size, n_units)`.

                - `batch_size` the number of samples.
                - `n_units` the total number of hidden neurons
                    (`n_neurons + out_features`).

        Returns:
            y_pred (jax.Array): the network prediction. When `batch_size=1`. Out shape is `(out_features)`. Otherwise, `(batch_size, out_features)`.
            h_state (jax.Array): the final hidden state. Output shape is `(batch_size, n_units)`.
        """
        if x.ndim != 2:
            raise ValueError(
                f"Unsupported dimensionality: '{x.shape}'. Should be 2 dimensional with: '(batch_size, features)'."
            )

        batch_size, features = x.shape

        if h_state is None:
            h_state = jnp.zeros((batch_size, self.hidden_size))

        # Batch -> (batch_size, out_features)
        x = self.act(self.inter(x))
        x, h_state = self.command(x, h_state)
        y_pred = self.motor(self.act(x))

        # Single item -> (out_features)
        if y_pred.shape[0] == 1:
            y_pred = y_pred.squeeze(0)

        # h_state -> (batch_size, n_units)
        return y_pred, h_state
