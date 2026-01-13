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

from typing import Self, Tuple

import chex
import jax.numpy as jnp
import numpy as np

from velora.config.spec import (
    ACMHeadSpec,
    HeadSpec,
    LayerSpec,
    NCPWiringSpec,
    OCMHeadSpec,
    SingleHeadSpec,
)


def _synapse_count(count: int, density_level: float, *, scale: int = 1) -> int:
    """
    Utility method that computes the synapse count for a single NCP layer.

    Parameters
    ----------
    count : int
        The number of neurons
    density_level : float
        The density of the layer connections (`1.0 - sparsity_level`)
    scale : int (optional)
        A scale factor. Default is `1`

    Returns
    -------
    count : int
        Synapse count
    """
    return max(int(count * density_level * scale), 1)


def _make_mask(
    shape: Tuple[int, int], count: int, rng: np.random.Generator
) -> chex.Array:
    """
    Randomly assigns connections to nodes by populating sparsity mask.

    Note -
        Performs two operations:

        1. Applies minimum connections (count) to all nodes
        2. Ensures all nodes have at least 1 connection

    Parameters
    ----------
    shape : Tuple[int, int]
        Mask shape `(n_inputs, n_outputs)`
    count : int
        Number of connections per node
    rng : np.random.Generator
        NumPy random number generator

    Returns
    -------
    mask : jax.Array
        Populated sparsity mask
    """
    n_nodes, n_cols = shape
    mask = np.zeros(shape, dtype=np.int32)

    # Add required connection count
    col_indices = rng.choice(n_cols, (n_nodes, count))
    polarities = rng.choice([-1, 1], jnp.shape(col_indices))
    row_indices = np.expand_dims(np.arange(n_nodes), 1)

    mask[row_indices, col_indices] = polarities

    # Add missing node connections (if applicable)
    # -> Every node in 'num_cols' must have at least 1 connection
    # -> Column with all 0s = non-connected node
    unconnected = np.where((mask == 0).all(axis=0))[0]
    if unconnected.size > 0:
        # For each missing connection, randomly select a node and add connection
        # -> row = node
        row_indices = rng.integers(0, n_nodes, (unconnected.size,))
        random_polarities = rng.choice([-1, 1], (unconnected.size,))
        mask[row_indices, unconnected] = random_polarities

    return jnp.asarray(mask)


def _build_layer(
    shape: Tuple[int, int],
    n_connections: int,
    rng: np.random.Generator,
) -> LayerSpec:
    """
    Build a single layer specification.

    Parameters
    ----------
    shape : Tuple[int, int]
        Mask shape `(n_inputs, n_outputs)`
    n_connections : int
        Number of connections per node
    rng : np.random.Generator
        NumPy random number generator

    Returns
    -------
    spec : LayerSpec
        An NCP layer specification
    """
    return LayerSpec(mask=_make_mask(shape, n_connections, rng), n_hidden=shape[1])


class NCPWiringBuilder:
    """
    Builder for NCP wiring specifications.

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Must be a value between `[0.1, 0.9]`:

        - Where `0.1` neurons are very dense
        - Where `0.9` neurons are very sparse

        Default is `0.5`
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        *,
        seed: int = 28,
        sparsity: float = 0.5,
    ) -> None:
        if not 0.1 <= sparsity <= 0.9:
            raise ValueError(f"'{sparsity=}' must be between '[0.1, 0.9]'.")

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.seed = seed
        self.density = 1.0 - sparsity

        self.n_command = max(int(0.4 * self.n_neurons), 1)
        self.n_inter = self.n_neurons - self.n_command

        self._head_spec: HeadSpec | None = None
        self._rng = np.random.default_rng(seed)

    def _motor_connection_count(self) -> int:
        """
        A utility method to compute the motor head connection count.

        Returns
        -------
        count : int
            Motor head connection count
        """
        return _synapse_count(self.n_command, self.density, scale=2)

    def with_acm_heads(self, *, z_dim: int, aux_pi_dim: int, q_dim: int) -> Self:
        """
        Configure wiring with ACM output heads.

        Parameters
        ----------
        z_dim : int
            Dimension of action-conditioned prediction head
        aux_pi_dim : int
            Dimension of auxiliary policy head
        q_dim : int
            Dimension of action-value head

        Returns
        -------
        self : Self
            Updated object with `_head_spec`
        """
        motor_conn = self._motor_connection_count()
        self._head_spec = ACMHeadSpec(
            z=_build_layer((self.n_command, z_dim), motor_conn, self._rng),
            aux_pi=_build_layer((self.n_command, aux_pi_dim), motor_conn, self._rng),
            q=_build_layer((self.n_command, q_dim), motor_conn, self._rng),
        )
        return self

    def with_ocm_heads(self, *, y_dim: int, pi_dim: int) -> Self:
        """
        Configure wiring with OCM output heads.

        Parameters
        ----------
        y_dim : int
            Dimension of observation-conditioned prediction head
        pi_dim : int
            Dimension of policy head
        """
        motor_conn = self._motor_connection_count()
        self._head_spec = OCMHeadSpec(
            y=_build_layer((self.n_command, y_dim), motor_conn, self._rng),
            pi=_build_layer((self.n_command, pi_dim), motor_conn, self._rng),
        )
        return self

    def with_single_head(self, *, out_dim: int) -> Self:
        """
        Configure wiring with a single output head.

        Parameters
        ----------
        out_dim : int
            Dimension of output head
        """
        motor_conn = self._motor_connection_count()
        self._head_spec = SingleHeadSpec(
            out=_build_layer((self.n_command, out_dim), motor_conn, self._rng),
        )
        return self

    def build(self) -> NCPWiringSpec:
        """
        Builds the NCP wiring specification.

        Returns
        -------
        wiring : NCPWiringSpec
            NCP wiring specification

        Raises
        ------
        heads_missing : ValueError
            Missing output heads. Resolved by calling a `with_..._heads()` method first.
        """
        if self._head_spec is None:
            raise ValueError(
                "No heads defined. Call a `with_..._heads()` method first."
            )

        # Connection counts
        inter_count = _synapse_count(self.n_inter, self.density)
        command_count = _synapse_count(self.n_command, self.density)

        # sensory -> inter
        inter = _build_layer((self.in_features, self.n_inter), inter_count, self._rng)
        # inter -> command
        command = _build_layer((self.n_inter, self.n_command), command_count, self._rng)

        return NCPWiringSpec(inter=inter, command=command, motor=self._head_spec)


def build_acm_wiring(
    in_features: int,
    n_neurons: int,
    z_dim: int,
    num_actions: int,
    q_dim: int = 1,
    *,
    seed: int = 28,
    sparsity_level: float = 0.5,
) -> NCPWiringSpec:
    """
    Creates NCP wiring for an Action-Conditional Model (ACM).

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    z_dim : int
        Dimension of action-conditioned prediction head
    num_actions : int
        Number of discrete actions
    q_dim : int (optional)
        Dimension of action-value prediction head.
        Default is `1` (scalar)
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity_level : float (optional)
        Controls the connection sparsity between neurons.
        Must be a value between `[0.1, 0.9]`:

        - Where `0.1` neurons are very dense
        - Where `0.9` neurons are very sparse

        Default is `0.5`

    Returns
    -------
    wiring : NCPWiringSpec
        NCP wiring
    """
    return (
        NCPWiringBuilder(in_features, n_neurons, seed=seed, sparsity=sparsity_level)
        .with_acm_heads(z_dim=z_dim, aux_pi_dim=num_actions, q_dim=q_dim)
        .build()
    )


def build_ocm_wiring(
    in_features: int,
    n_neurons: int,
    y_dim: int,
    num_actions: int,
    *,
    seed: int = 28,
    sparsity_level: float = 0.5,
) -> NCPWiringSpec:
    """
    Creates NCP wiring for an Observation-Conditional Model (OCM).

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    y_dim : int
        Dimension of observation-conditioned prediction head
    num_actions : int
        Number of discrete actions
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity_level : float (optional)
        Controls the connection sparsity between neurons.
        Must be a value between `[0.1, 0.9]`:

        - Where `0.1` neurons are very dense
        - Where `0.9` neurons are very sparse

        Default is `0.5`

    Returns
    -------
    wiring : NCPWiringSpec
        NCP wiring
    """
    return (
        NCPWiringBuilder(in_features, n_neurons, seed=seed, sparsity=sparsity_level)
        .with_ocm_heads(y_dim=y_dim, pi_dim=num_actions)
        .build()
    )


def build_ncp_wiring(
    in_features: int,
    n_neurons: int,
    out_features: int,
    *,
    seed: int = 28,
    sparsity_level: float = 0.5,
) -> NCPWiringSpec:
    """
    Creates NCP wiring with a single output head.

    Note -
        NCPs have three layers:

        1. Inter (input)
        2. Command (hidden)
        3. Motor (output)

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    out_features : int
        Number of outputs (motor nodes)
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity_level : float (optional)
        Controls the connection sparsity between neurons.
        Must be a value between `[0.1, 0.9]`:

        - When `0.1` neurons are very dense
        - When `0.9` neurons are very sparse

        Default is `0.5`

    Returns
    -------
    wiring : NCPWiringSpec
        NCP wiring
    """
    return (
        NCPWiringBuilder(in_features, n_neurons, seed=seed, sparsity=sparsity_level)
        .with_single_head(out_dim=out_features)
        .build()
    )
