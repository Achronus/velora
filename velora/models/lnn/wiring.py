from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass(frozen=True)
class NCPMaskWithCounts:
    """
    A storage container for a single Neural Circuit Policy (NCP) layer.
    Includes it's sparsity mask, neuron counts and synapse connection count.

    Parameters:
        mask (jax.Array): sparse weight mask for NCP layer
        n_hidden (int): number of NCP neuron nodes
        n_connections (int): number of synapse connections (weights)
    """

    mask: chex.Array
    n_hidden: int
    n_connections: int


@struct.dataclass
class NCPWiring:
    """
    A storage container for a Neural Circuit Policy (NCP) wiring.

    Parameters:
        inter (NCPMaskWithCounts): inter layer details
        command (NCPMaskWithCounts): command layer details
        motor (NCPMaskWithCounts): motor layer details
    """

    inter: NCPMaskWithCounts
    command: NCPMaskWithCounts
    motor: NCPMaskWithCounts


@struct.dataclass(frozen=True)
class HeadConfig:
    """
    A base configuration for NCP network output heads that must be
    inherited from when used in the `build_multi_head_ncp_wiring()` method.
    """

    def hidden_count(self) -> int:
        """Returns the total hidden node count."""
        raise NotImplementedError()


@struct.dataclass(frozen=True)
class ACMHeadConfig(HeadConfig):
    """
    Head configuration for an Action-Conditional Model (ACM) network.

    Parameters:
        z (NCPMaskWithCounts): action-conditioned prediction head details
        aux_pi (NCPMaskWithCounts): auxiliary policy prediction head details
        q (NCPMaskWithCounts): action-value prediction head details
    """

    z: NCPMaskWithCounts
    aux_pi: NCPMaskWithCounts
    q: NCPMaskWithCounts

    def hidden_count(self) -> int:
        return self.z.n_hidden + self.aux_pi.n_hidden + self.q.n_hidden


@struct.dataclass(frozen=True)
class OCMHeadConfig(HeadConfig):
    """
    Head configuration for an Observation-Conditional Model (OCM) network.

    Parameters:
        y (NCPMaskWithCounts): observation-conditioned prediction head details
        pi (NCPMaskWithCounts): policy head details
    """

    y: NCPMaskWithCounts
    pi: NCPMaskWithCounts

    def hidden_count(self) -> int:
        return self.y.n_hidden + self.pi.n_hidden


@struct.dataclass
class NCPWiringMultiHead:
    """
    A storage container for a Neural Circuit Policy (NCP) wiring with multiple output heads.

    Parameters:
        inter (NCPMaskWithCounts): inter layer details
        command (NCPMaskWithCounts): command layer details
        motor (HeadConfig): motor layer details
    """

    inter: NCPMaskWithCounts
    command: NCPMaskWithCounts
    motor: HeadConfig


def build_ncp_wiring(
    in_features: int,
    n_neurons: int,
    out_features: int,
    *,
    seed: int = 28,
    sparsity_level: float = 0.5,
) -> NCPWiring:
    """
    Creates sparse wiring masks with neuron counts for
    a Neural Circuit Policy (NCP) Network.

    !!! note

        NCPs have three layers:

        1. Inter (input)
        2. Command (hidden)
        3. Motor (output)

    Parameters:
        in_features (int): number of inputs (sensory nodes)
        n_neurons (int): number of decision nodes (inter + command nodes)
        out_features (int): number of outputs (motor nodes)
        seed (int, optional): random number generator seed
        sparsity_level (float, optional): controls the connection sparsity between neurons.

            Must be a value between `[0.1, 0.9]` -

            - When `0.1` neurons are very dense.
            - When `0.9` neurons are very sparse.
    """
    if sparsity_level < 0.1 or sparsity_level > 0.9:
        raise ValueError(f"'{sparsity_level=}' must be between '[0.1, 0.9]'.")

    density_level = 1.0 - sparsity_level
    n_inter_and_command = n_neurons - out_features

    n_command = max(int(0.4 * n_inter_and_command), 1)
    n_inter = n_inter_and_command - n_command

    inter_count = synapse_count(n_inter, density_level)
    command_count = synapse_count(n_command, density_level)
    motor_count = synapse_count(n_command, density_level, scale=2)

    # sensory -> inter
    inter = build_mask_with_counts((in_features, n_inter), inter_count, seed)

    # inter -> command
    command = build_mask_with_counts((n_inter, n_command), command_count, seed)

    # command -> motor
    motor = build_mask_with_counts((n_command, out_features), motor_count, seed)

    return NCPWiring(inter=inter, command=command, motor=motor)


def build_acm_wiring(
    in_features: int,
    n_neurons: int,
    z_dim: int,
    num_actions: int,
    q_dim: int = 1,
    *,
    seed: int = 28,
    sparsity_level: float = 0.5,
) -> NCPWiringMultiHead:
    """
    Creates NCP wiring for an Action-Conditional Model (ACM).

    Parameters:
        in_features (int): number of inputs (sensory nodes)
        n_neurons (int): number of decision nodes (inter + command nodes)
        head_sizes (ACMHeadConfig): an object containing head details and their
            counts (motor nodes)
        z_dim (int): dimension of action-conditioned prediction head
        num_actions (int): number of discrete actions
        q_dim (int, optional): dimension of action-value prediction head.
            Default is `1` (scalar)
        seed (int, optional): random number generator seed
        sparsity_level (float, optional): controls the connection sparsity between neurons.

            Must be a value between `[0.1, 0.9]` -

            - Where `0.1` neurons are very dense.
            - Where `0.9` neurons are very sparse.

    Returns:
        wiring (NCPWiringMultiHead): wiring with `ACMHeadConfig` motor
    """
    if sparsity_level < 0.1 or sparsity_level > 0.9:
        raise ValueError(f"'{sparsity_level=}' must be between '[0.1, 0.9]'.")

    density_level = 1.0 - sparsity_level

    n_command = max(int(0.4 * n_neurons), 1)
    n_inter = n_neurons - n_command

    inter_count = synapse_count(n_inter, density_level)
    command_count = synapse_count(n_command, density_level)
    motor_count = synapse_count(n_command, density_level, scale=2)

    # sensory -> inter
    inter = build_mask_with_counts((in_features, n_inter), inter_count, seed)

    # inter -> command
    command = build_mask_with_counts((n_inter, n_command), command_count, seed)

    # command -> motor
    motor = ACMHeadConfig(
        z=build_mask_with_counts((n_command, z_dim), motor_count, seed),
        aux_pi=build_mask_with_counts((n_command, num_actions), motor_count, seed),
        q=build_mask_with_counts((n_command, q_dim), motor_count, seed),
    )

    return NCPWiringMultiHead(inter=inter, command=command, motor=motor)


def build_ocm_wiring(
    in_features: int,
    n_neurons: int,
    y_dim: int,
    num_actions: int,
    *,
    seed: int = 28,
    sparsity_level: float = 0.5,
) -> NCPWiringMultiHead:
    """
    Creates NCP wiring for an Observation-Conditional Model (OCM).

    Parameters:
        in_features (int): number of inputs (sensory nodes)
        n_neurons (int): number of decision nodes (inter + command nodes)
        head_sizes (ACMHeadConfig): an object containing head details and their
            counts (motor nodes)
        y_dim (int): dimension of observation-conditioned prediction head
        num_actions (int): number of discrete actions
        seed (int, optional): random number generator seed
        sparsity_level (float, optional): controls the connection sparsity between neurons.

            Must be a value between `[0.1, 0.9]` -

            - Where `0.1` neurons are very dense.
            - Where `0.9` neurons are very sparse.

    Returns:
        wiring (NCPWiringMultiHead): wiring with `OCMHeadConfig` motor
    """
    if sparsity_level < 0.1 or sparsity_level > 0.9:
        raise ValueError(f"'{sparsity_level=}' must be between '[0.1, 0.9]'.")

    density_level = 1.0 - sparsity_level

    n_command = max(int(0.4 * n_neurons), 1)
    n_inter = n_neurons - n_command

    inter_count = synapse_count(n_inter, density_level)
    command_count = synapse_count(n_command, density_level)
    motor_count = synapse_count(n_command, density_level, scale=2)

    # sensory -> inter
    inter = build_mask_with_counts((in_features, n_inter), inter_count, seed)

    # inter -> command
    command = build_mask_with_counts((n_inter, n_command), command_count, seed)

    # command -> motor
    motor = OCMHeadConfig(
        y=build_mask_with_counts((n_command, y_dim), motor_count, seed),
        pi=build_mask_with_counts((n_command, num_actions), motor_count, seed),
    )

    return NCPWiringMultiHead(inter=inter, command=command, motor=motor)


def synapse_count(count: int, density_level: float, *, scale: int = 1) -> int:
    """
    Utility method. Computes the synapse count for a single NCP layer.

    Parameters:
        count (int): the number of neurons
        density_level (float): the density of the layer connections
            (`1.0 - sparsity_level`)
        scale (int, optional): a scale factor

    Returns:
        count (int): synapse count.
    """
    return max(int(count * density_level * scale), 1)


def make_mask(shape: Tuple[int, int], count: int, seed: int) -> chex.Array:
    """
    Randomly assigns connections to nodes by populating sparsity mask.

    !!! note "Performs two operations"

        1. Applies minimum connections (count) to all nodes.
        2. Ensures all nodes have at least 1 connection.

    Parameters:
        shape (Tuple[int, int]): mask shape `(n_inputs, n_outputs)`
        count (int): number of connections per node
        seed (int): random number generator seed

    Returns:
        mask (jax.Array): populated sparsity mask
    """
    rng = np.random.default_rng(seed)

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
    is_col_all_zero = (mask == 0).all(axis=0)
    col_zero_indices = np.nonzero(is_col_all_zero)[0]
    zero_count = col_zero_indices.size

    if zero_count > 0:
        # For each missing connection, randomly select a node and add connection
        # -> row = node
        row_indices = rng.integers(0, n_nodes, (zero_count,))
        random_polarities = rng.choice([-1, 1], (zero_count,))
        mask[row_indices, col_zero_indices] = random_polarities

    return jnp.asarray(mask)


def build_mask_with_counts(
    shape: Tuple[int, int],
    n_connections: int,
    seed: int,
) -> NCPMaskWithCounts:
    """
    Helper method. Build a single `NCPMaskWithCounts` object.

    Parameters:
        shape (Tuple[int, int]): mask shape `(n_inputs, n_outputs)`
        n_connections (int): number of connections per node
        seed (int): random number generator seed

    Returns:
        ncp_mask_with_counts (NCPMaskWithCounts): a container with the mask,
            neuron count and synapse connection count
    """
    mask = make_mask(shape, n_connections, seed)
    n_hidden = shape[1]

    return NCPMaskWithCounts(
        mask=mask,
        n_hidden=n_hidden,
        n_connections=n_connections,
    )
