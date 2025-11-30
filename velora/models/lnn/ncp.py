from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from velora.constants import DEFAULT_HIDDEN_INIT
from velora.models.lnn.cell import NCPLiquidCell
from velora.utils.nn import active_parameters, total_parameters
from velora.wiring import build_ncp_wiring


class LiquidNCPNetwork(nnx.Module):
    """
    A CfC Liquid Neural Circuit Policy (NCP) Network with three layers:

    1. Inter (input) - a `NCPLiquidCell` layer
    2. Command (hidden) - a `NCPLiquidCell` layer
    3. Motor (output) - a `NCPLiquidCell` layer

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
        self.hidden_size = self.n_neurons

        self.wiring = nnx.data(
            build_ncp_wiring(
                in_features,
                n_neurons,
                out_features,
                seed=seed,
                sparsity_level=sparsity_level,
            )
        )

        self.inter = NCPLiquidCell(
            in_features,
            self.wiring.inter.n_nodes,
            self.wiring.inter.mask,
            rngs=self.rngs,
            init_type=init_type,
        )

        self.command = NCPLiquidCell(
            self.wiring.inter.n_nodes,
            self.wiring.command.n_nodes,
            self.wiring.command.mask,
            rngs=self.rngs,
            init_type=init_type,
        )

        self.motor = NCPLiquidCell(
            self.wiring.command.n_nodes,
            self.wiring.motor.n_nodes,
            self.wiring.motor.mask,
            rngs=self.rngs,
            init_type=init_type,
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

    def _split_h_state(
        self, h: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Helper method. Splits the NCPs hidden state into layer-specific states.

        Parameters:
            h (jax.Array): the network hidden state

        Returns:
            h_split (Tuple[chex.Array, chex.Array, chex.Array]): hidden state split
            into layers `(inter, command, motor)`
        """
        split_indices = jnp.cumsum(
            jnp.array([self.wiring.inter.n_nodes, self.wiring.command.n_nodes])
        )
        h_inter, h_command, h_motor = jnp.split(h, split_indices, axis=1)
        return h_inter, h_command, h_motor

    def __call__(
        self,
        x: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Performs a forward pass through the network.

        Parameters:
            x (jax.Array): an input tensor of shape: `(F, T)` or `(B, F, T)`.

                - `batch_size (B)` the number of samples per timestep.
                - `features (F)` the features at each timestep (e.g.,
                image features, joint coordinates, word embeddings, raw amplitude
                values).
                - `seq_length (T)` the number of sequences (e.g., trajectories,
                channels).
            h_state (jax.Array, optional): initial hidden state of the RNN with
                shape: `(B, H)`.

                - `batch_size (B)` the number of samples.
                - `n_units (H)` the total number of hidden neurons
                    (`n_neurons + out_features`).

            timespans (jax.Array, optional): time elapsed since previous timestep.
                For fixed intervals set to `None`. For varying timesteps shape
                should be `(T,)`
        Returns:
            y_pred (jax.Array): the network prediction. Shape `(B, F, T)`.
            h_state (jax.Array): the final hidden state. Shape `(B, H)`.
        """
        if x.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D input, got shape {jnp.shape(x)}")

        if x.ndim == 2:
            x = jnp.expand_dims(x, 0)

        B, F, T = jnp.shape(x)

        if h_state is None:
            h_state = jnp.zeros((B, self.hidden_size))

        timespans = jnp.ones(T) if timespans is None else timespans

        def _step(
            h: Tuple[chex.Array, chex.Array, chex.Array],
            inputs: Tuple[chex.Array, chex.Array],
        ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
            """Single step function."""
            h_inter, h_command, h_motor = h
            x_t, ts_t = inputs  # x_t -> (B, F), ts_t -> scalar

            # Forward through each liquid layer
            x_t, new_h_inter = self.inter(x_t, h_inter, ts_t)
            x_t, new_h_command = self.command(x_t, h_command, ts_t)
            y_t, new_h_motor = self.motor(x_t, h_motor, ts_t)  # y_t -> (B, F)

            new_h = (new_h_inter, new_h_command, new_h_motor)
            return new_h, y_t

        # x -> (T, B, F) for scanning over time dimension
        x_transposed = jnp.transpose(x, (2, 0, 1))

        # Split hidden states per layer
        h_split = self._split_h_state(h_state)
        scan_inputs = (x_transposed, timespans)

        new_h, y_pred = jax.lax.scan(_step, h_split, scan_inputs, length=T)

        h_state = jnp.concatenate(new_h, axis=1)
        y_pred = jnp.transpose(y_pred, (1, 2, 0))

        # y_pred, h_state -> (B, F, T), (B, H)
        return y_pred, h_state
