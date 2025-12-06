import chex
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class NCPMaskWithCounts:
    """
    A storage container for a single Neural Circuit Policy (NCP) layer.
    Includes it's sparsity mask, neuron counts and synapse connection count.

    Parameters:
        mask (jax.Array): sparse weight mask for NCP layer
        n_nodes (int): number of NCP neuron nodes
        n_connections (int): number of synapse connections (weights)
    """

    mask: chex.Array
    n_nodes: int
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
        n_neurons (int): number of decision nodes (inter and command nodes)
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

    inter_mask = make_mask(
        np.zeros((in_features, n_inter), dtype=np.int32),
        inter_count,
        seed,
    )  # sensory -> inter
    command_mask = make_mask(
        np.zeros((n_inter, n_command), dtype=np.int32),
        command_count,
        seed,
    )  # inter -> command
    motor_mask = make_mask(
        np.zeros((n_command, out_features), dtype=np.int32),
        motor_count,
        seed,
    )  # command -> motor

    return NCPWiring(
        inter=NCPMaskWithCounts(
            mask=jnp.asarray(inter_mask),
            n_nodes=n_inter,
            n_connections=inter_count,
        ),
        command=NCPMaskWithCounts(
            mask=jnp.asarray(command_mask),
            n_nodes=n_command,
            n_connections=command_count,
        ),
        motor=NCPMaskWithCounts(
            mask=jnp.asarray(motor_mask),
            n_nodes=out_features,
            n_connections=motor_count,
        ),
    )


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


def make_mask(mask: np.ndarray, count: int, seed: int) -> np.ndarray:
    """
    Randomly assigns connections to a set of nodes by populating
    its sparsity mask.

    !!! note "Performs two operations"

        1. Applies minimum connections (count) to all nodes.
        2. Checks all nodes have at least 1 connection.
            If not, adds a connection to 'missing' nodes.

    Parameters:
        mask (np.Array): the initialized mask
        count (int): number of connections per node
        seed (int): random number generator seed
    """
    rng = np.random.default_rng(seed)

    n_nodes, n_cols = jnp.shape(mask)

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

    return mask
