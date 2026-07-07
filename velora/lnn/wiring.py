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

from typing import Dict, NamedTuple, Tuple

import numpy as np
import torch


class Wiring(NamedTuple):
    """
    NCP wiring masks for a CfC-LNN network.

    All masks use the PyTorch layout `(out_features, in_features)` with
    connection polarities `(-1, 0, 1)`. Layer sizes are derivable from
    mask shapes, e.g. `inter.shape[0]` is the number of inter neurons.

    Parameters
    ----------
    inter : torch.Tensor
        Inter (input) layer mask of shape `(n_inter, in_features)`
    command : torch.Tensor
        Command (hidden) layer mask of shape `(n_command, n_inter)`
    heads : Dict[str, torch.Tensor]
        Motor (output) head masks keyed by head name, each of shape
        `(head_dim, n_command)`. Order matches the `heads` argument
        given to `build_wiring`
    """

    inter: torch.Tensor
    command: torch.Tensor
    heads: Dict[str, torch.Tensor]


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
    shape: Tuple[int, int],
    count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
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
    mask : torch.Tensor
        Populated sparsity mask, transposed to the PyTorch layout
        `(n_outputs, n_inputs)`
    """
    n_nodes, n_cols = shape
    mask = np.zeros(shape, dtype=np.int32)

    # Add required connection count
    col_indices = rng.choice(n_cols, (n_nodes, count))
    polarities = rng.choice([-1, 1], np.shape(col_indices))
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

    return torch.asarray(mask).T


def build_wiring(
    in_features: int,
    n_neurons: int,
    heads: Dict[str, int],
    *,
    seed: int = 28,
    sparsity: float = 0.5,
) -> Wiring:
    """
    Build the sparsity masks for a CfC-LNN NCP network.

    Splits `n_neurons` into inter and command nodes using:

    ```python
    n_command = max(int(0.4 * n_neurons), 1)
    n_inter = n_neurons - n_command
    ```

    Then wires three layer groups in sequence:

    1. Inter (input) - `sensory -> inter`
    2. Command (hidden) - `inter -> command`
    3. Motor (output) - `command -> head`, one per entry in `heads`

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    heads : Dict[str, int]
        Motor head output sizes keyed by head name,
        e.g. `{"pi": 64, "y": 8}`
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Must be a value between `[0.1, 0.9]`:

        - Where `0.1` neurons are very dense
        - Where `0.9` neurons are very sparse

        Default is `0.5`

    Returns
    -------
    wiring : Wiring
        The populated NCP wiring masks

    Raises
    ------
    invalid_sparsity : ValueError
        When `sparsity` is outside `[0.1, 0.9]`
    heads_missing : ValueError
        When `heads` is empty

    Examples
    --------
    ```python
    wiring = build_wiring(4, 16, {"out": 2})
    wiring.inter.shape  # (10, 4)
    wiring.command.shape  # (6, 10)
    wiring.heads["out"].shape  # (2, 6)
    ```
    """
    if not 0.1 <= sparsity <= 0.9:
        raise ValueError(f"'{sparsity=}' must be between '[0.1, 0.9]'.")

    if not heads:
        raise ValueError("'heads' must contain at least one entry.")

    density = 1.0 - sparsity
    n_command = max(int(0.4 * n_neurons), 1)
    n_inter = n_neurons - n_command
    rng = np.random.default_rng(seed)

    inter = _make_mask(
        (in_features, n_inter),
        _synapse_count(n_inter, density),
        rng,
    )
    command = _make_mask(
        (n_inter, n_command),
        _synapse_count(n_command, density),
        rng,
    )

    motor_count = _synapse_count(n_command, density, scale=2)
    head_masks = {
        name: _make_mask((n_command, dim), motor_count, rng)
        for name, dim in heads.items()
    }

    return Wiring(inter=inter, command=command, heads=head_masks)
