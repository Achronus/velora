from typing import Dict, Literal, Optional, Tuple, get_args

import chex
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from velora.models.lnn.cell import NCPLiquidCell
from velora.models.lnn.wiring import build_multi_head_ncp_wiring
from velora.utils.nn import active_parameters, total_parameters

ACMHeads = Literal["z", "aux_pi", "q"]
PMHeads = Literal["pi", "y"]


class ACM(nnx.Module):
    """
    An Action-Conditional Model that uses a Liquid Neural Network (LNN)
    with 3 output heads:

        1. Action-conditioned prediction: z(s, a)
        2. Auxiliary policy prediction: p(s, a)
        3. Action-value: q(s, a)

    Parameters:
        in_features (int): number of inputs (sensory nodes)
        n_neurons (int): number of decision nodes (inter and command nodes)
        head_sizes (Dict[str, int]): a list of head names and their number of out
            features (motor nodes).
            Must match: `{"z": [int], "aux_pi": [int], "q": [int]}`
        seed (int, optional): random number generator seed. Default is `28`
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        head_sizes: Dict[str | ACMHeads, int],
        *,
        seed: int = 28,
    ) -> None:
        validate_head_sizes = set(get_args(ACMHeads)) & set(head_sizes.keys())
        if len(validate_head_sizes) != 3:
            raise ValueError(
                f"`head_sizes` must have values with the keys: {get_args(ACMHeads)}."
            )

        self.n_actions = head_sizes["aux_pi"]

        self.in_features = in_features + self.n_actions
        self.n_neurons = n_neurons
        self.seed = seed
        self.rngs = nnx.Rngs(params=seed)

        self.wiring = nnx.data(
            build_multi_head_ncp_wiring(
                self.in_features,
                n_neurons,
                head_sizes,
                seed=seed,
            )
        )

        self.head_sizes = head_sizes
        self.hidden_size = (
            self.wiring.inter.n_nodes
            + self.wiring.command.n_nodes
            + sum(motor.n_nodes for motor in self.wiring.motor.values())
        )

        self.inter = NCPLiquidCell(
            self.in_features,
            self.wiring.inter.n_nodes,
            self.wiring.inter.mask,
            rngs=self.rngs,
        )

        self.command = NCPLiquidCell(
            self.wiring.inter.n_nodes,
            self.wiring.command.n_nodes,
            self.wiring.command.mask,
            rngs=self.rngs,
        )

        self.z_head = NCPLiquidCell(
            self.wiring.command.n_nodes,
            self.wiring.motor["z"].n_nodes,
            self.wiring.motor["z"].mask,
            rngs=self.rngs,
        )
        self.aux_pi_head = NCPLiquidCell(
            self.wiring.command.n_nodes,
            self.wiring.motor["aux_pi"].n_nodes,
            self.wiring.motor["aux_pi"].mask,
            rngs=self.rngs,
        )
        self.q_head = NCPLiquidCell(
            self.wiring.command.n_nodes,
            self.wiring.motor["q"].n_nodes,
            self.wiring.motor["q"].mask,
            rngs=self.rngs,
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
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Helper method. Splits the NCPs hidden state into layer-specific states.

        Parameters:
            h (jax.Array): the network hidden state

        Returns:
            h_split (Tuple[chex.Array, ...]): hidden state split
            into layers `(inter, command, z, aux, q)`
        """
        split_indices = jnp.cumsum(
            jnp.array(
                [
                    self.wiring.inter.n_nodes,
                    self.wiring.command.n_nodes,
                    self.wiring.motor["z"].n_nodes,
                    self.wiring.motor["aux_pi"].n_nodes,
                ]
            )
        )
        h_inter, h_command, h_z, h_aux, h_q = jnp.split(h, split_indices, axis=1)
        return h_inter, h_command, h_z, h_aux, h_q

    def encode_obs_with_actions(self, x: chex.Array) -> chex.Array:
        """
        Expands the input observation with one-hot encoded actions (A)
        using an identity matrix for all batches.

        Parameters:
            x (jax.Array): the batch of input observations in the shape
                `(B, F, T)`

        Returns:
            x_new (jax.Array): obs with batched one-hot encoded actions
                in the shape `(BA, F+A, T)`
        """
        B, F, T = jnp.shape(x)

        # Expand state for all actions: (B, F, T) -> (B*A, F, T)
        x_expanded = jnp.repeat(x, self.n_actions, axis=0)

        # Create one-hot actions: (BA, A) -> (BA, A, T)
        one_hot_actions = jnp.eye(self.n_actions)  # (A, A)
        one_hot_actions = jnp.tile(one_hot_actions, [B, 1])  # (BA, A)
        one_hot_actions = jnp.expand_dims(one_hot_actions, -1)  # (BA, A, 1)
        one_hot_actions = jnp.tile(one_hot_actions, [1, 1, T])  # (BA, A, T)

        return jnp.concatenate([x_expanded, one_hot_actions], axis=1)  # (BA, F+A, T)

    def __call__(
        self,
        x: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Performs a forward pass through the network.

        Parameters:
            x (jax.Array): an input array of shape: `(F, T)` or `(B, F, T)`.

                - `batch_size (B)` the number of samples per timestep.
                - `features (F)` the features at each timestep
                - `seq_length (T)` the number of sequences (e.g., trajectories).
            h_state (jax.Array, optional): initial hidden state of the RNN with
                shape: `(B, H)`.

                - `batch_size (B)` the number of samples per timestep.
                - `n_units (H)` the total number of hidden neurons
                    (`n_neurons + out_features`).

            timespans (jax.Array, optional): time elapsed since previous timestep.
                For fixed intervals set to `None`. For varying timesteps shape
                should be `(T,)`
        Returns:
            z (jax.Array): the action-conditioned prediction. Shape `(B, A, F, T)`.
            aux_pi (jax.Array): the auxiliary policy prediction. Shape `(B, A, F, T)`.
            q (jax.Array): the action-value prediction. Shape `(B, A, F, T)`.
            h_state (jax.Array): the final hidden state. Shape `(BA, H)`.
        """
        if x.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D input, got shape {jnp.shape(x)}")

        if x.ndim == 2:
            x = jnp.expand_dims(x, 0)

        B, F, T = jnp.shape(x)
        x = self.encode_obs_with_actions(x)  # (BA, F+A, T)

        if h_state is None:
            h_state = jnp.zeros((B * self.n_actions, self.hidden_size))  # (BA, H)

        timespans = jnp.ones(T) if timespans is None else timespans

        def _step(
            h: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
            inputs: Tuple[chex.Array, chex.Array],
        ) -> Tuple[
            Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array, chex.Array],
        ]:
            """Single step function."""
            h_inter, h_command, h_z, h_aux, h_q = h
            x_t, ts_t = inputs  # x_t -> (BA, F+A), ts_t -> scalar

            # Forward through each liquid layer
            x_t, new_h_inter = self.inter(x_t, h_inter, ts_t)
            x_t, new_h_command = self.command(x_t, h_command, ts_t)

            z_t, new_h_z = self.z_head(x_t, h_z, ts_t)  # z_t -> (B, F)
            aux_t, new_h_aux = self.aux_pi_head(x_t, h_aux, ts_t)  # aux_t -> (B, F)
            q_t, new_h_q = self.q_head(x_t, h_q, ts_t)  # q_t -> (B, F)

            new_h = (new_h_inter, new_h_command, new_h_z, new_h_aux, new_h_q)
            preds = (z_t, aux_t, q_t)
            return new_h, preds

        # x -> (T, BA, F+A) for scanning over time dimension
        x_transposed = jnp.transpose(x, (2, 0, 1))

        # Split hidden states per layer
        h_split = self._split_h_state(h_state)
        scan_inputs = (x_transposed, timespans)

        new_h, preds = jax.lax.scan(_step, h_split, scan_inputs, length=T)

        h_state = jnp.concatenate(new_h, axis=1)  # (BA, H)
        z, aux_pi, q = preds

        # Transpose from (T, BA, F) -> (BA, F, T)
        z = jnp.transpose(z, (1, 2, 0))
        aux_pi = jnp.transpose(aux_pi, (1, 2, 0))
        q = jnp.transpose(q, (1, 2, 0))

        # Reshape from (BA, F, T) -> (B, A, F, T)
        z = z.reshape(B, self.n_actions, -1, T)
        aux_pi = aux_pi.reshape(B, self.n_actions, -1, T)
        q = q.reshape(B, self.n_actions, -1, T)

        return z, aux_pi, q, h_state


class PolicyNet(nnx.Module):
    """
    A Policy Network for the DiscoRL update rule that uses a
    Liquid Neural Network (LNN) with 2 output heads:

        1. Policy: π(s, a)
        2. Observation-conditioned prediction: y(s)

    Parameters:
        in_features (int): number of inputs (sensory nodes)
        n_neurons (int): number of decision nodes (inter and command nodes)
        head_sizes (Dict[str, int]): a list of head names and their number of out
            features (motor nodes). Must match: `{"pi": [int], "y": [int]}`
        seed (int, optional): random number generator seed. Default is `28`
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        head_sizes: Dict[str | PMHeads, int],
        *,
        seed: int = 28,
    ) -> None:
        validate_head_sizes = set(get_args(PMHeads)) & set(head_sizes.keys())
        if len(validate_head_sizes) != 2:
            raise ValueError(
                f"`head_sizes` must have values with the keys: {get_args(PMHeads)}."
            )

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.seed = seed
        self.rngs = nnx.Rngs(params=seed)

        self.wiring = nnx.data(
            build_multi_head_ncp_wiring(
                self.in_features,
                n_neurons,
                head_sizes,
                seed=seed,
            )
        )

        self.head_sizes = head_sizes
        self.hidden_size = (
            self.wiring.inter.n_nodes
            + self.wiring.command.n_nodes
            + sum(motor.n_nodes for motor in self.wiring.motor.values())
        )

        self.inter = NCPLiquidCell(
            self.in_features,
            self.wiring.inter.n_nodes,
            self.wiring.inter.mask,
            rngs=self.rngs,
        )

        self.command = NCPLiquidCell(
            self.wiring.inter.n_nodes,
            self.wiring.command.n_nodes,
            self.wiring.command.mask,
            rngs=self.rngs,
        )

        self.pi_head = NCPLiquidCell(
            self.wiring.command.n_nodes,
            self.wiring.motor["pi"].n_nodes,
            self.wiring.motor["pi"].mask,
            rngs=self.rngs,
        )
        self.y_head = NCPLiquidCell(
            self.wiring.command.n_nodes,
            self.wiring.motor["y"].n_nodes,
            self.wiring.motor["y"].mask,
            rngs=self.rngs,
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
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Helper method. Splits the NCPs hidden state into layer-specific states.

        Parameters:
            h (jax.Array): the network hidden state

        Returns:
            h_split (Tuple[chex.Array, ...]): hidden state split
            into layers `(inter, command, pi, y, )`
        """
        split_indices = jnp.cumsum(
            jnp.array(
                [
                    self.wiring.inter.n_nodes,
                    self.wiring.command.n_nodes,
                    self.wiring.motor["pi"].n_nodes,
                ]
            )
        )
        h_inter, h_command, h_pi, h_y = jnp.split(h, split_indices, axis=1)
        return h_inter, h_command, h_pi, h_y

    def __call__(
        self,
        x: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Performs a forward pass through the network.

        Parameters:
            x (jax.Array): an input array of shape: `(F, T)` or `(B, F, T)`.

                - `batch_size (B)` the number of samples per timestep.
                - `features (F)` the features at each timestep
                - `seq_length (T)` the number of sequences (e.g., trajectories).
            h_state (jax.Array, optional): initial hidden state of the RNN with
                shape: `(B, H)`.

                - `batch_size (B)` the number of samples per timestep.
                - `n_units (H)` the total number of hidden neurons
                    (`n_neurons + out_features`).

            timespans (jax.Array, optional): time elapsed since previous timestep.
                For fixed intervals set to `None`. For varying timesteps shape
                should be `(T,)`
        Returns:
            pi (jax.Array): the policy prediction. Shape `(B, F, T)`.
            y (jax.Array): the observation-conditioned prediction.
                Shape `(B, F, T)`.
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
            h: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
            inputs: Tuple[chex.Array, chex.Array],
        ) -> Tuple[
            Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array],
        ]:
            """Single step function."""
            h_inter, h_command, h_pi, h_y = h
            x_t, ts_t = inputs  # x_t -> (B, F), ts_t -> scalar

            # Forward through each liquid layer
            x_t, new_h_inter = self.inter(x_t, h_inter, ts_t)
            x_t, new_h_command = self.command(x_t, h_command, ts_t)

            pi_t, new_h_pi = self.pi_head(x_t, h_pi, ts_t)  # pi_t -> (B, F)
            y_t, new_h_y = self.y_head(x_t, h_y, ts_t)  # y_t -> (B, F)

            new_h = (new_h_inter, new_h_command, new_h_pi, new_h_y)
            preds = (pi_t, y_t)
            return new_h, preds

        # x -> (T, B, F) for scanning over time dimension
        x_transposed = jnp.transpose(x, (2, 0, 1))

        # Split hidden states per layer
        h_split = self._split_h_state(h_state)
        scan_inputs = (x_transposed, timespans)

        new_h, preds = jax.lax.scan(_step, h_split, scan_inputs, length=T)

        h_state = jnp.concatenate(new_h, axis=1)  # (B, H)
        pi, y = preds

        # Transpose from (T, B, F) -> (B, F, T)
        pi = jnp.transpose(pi, (1, 2, 0))
        y = jnp.transpose(y, (1, 2, 0))

        return pi, y, h_state
