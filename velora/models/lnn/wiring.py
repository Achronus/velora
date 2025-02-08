from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pydantic import BaseModel

import torch


class NeuronCounts(BaseModel):
    """Storage container for layer counts."""

    sensory: int
    inter: int
    command: int
    motor: int


class ConnectionCounts(BaseModel):
    """Storage container for synapse connection counts."""

    sensory: int
    inter: int
    command: int
    motor: int


@dataclass
class LayerMasks:
    """Storage container for layer masks."""

    inter: torch.Tensor
    command: torch.Tensor
    motor: torch.Tensor


class WiringConfig(BaseModel):
    """A model for extracting wiring config settings."""

    n_neurons: NeuronCounts
    n_connections: ConnectionCounts
    density_level: float


class Wiring:
    """
    Wiring for a Neural Circuit Policy (NCP).

    Parameters:
        in_features (int): number of inputs (sensory nodes)
        n_neurons (int): number of decision nodes (inter and command nodes)
        out_features (int): number of out features (motor nodes)
        sparsity_level (float, optional): controls the connection sparsity between
            neurons. Must be a value between `[0.1, 0.9]`. When `0.1` neurons are
            very dense, when `0.9` they are very sparse. Default is '0.5'
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        sparsity_level: float = 0.5,
    ) -> None:
        if sparsity_level < 0.1 or sparsity_level > 0.9:
            raise ValueError(f"'{sparsity_level=}' must be between '[0.1, 0.9]'.")

        self.density_level = 1.0 - sparsity_level
        self.units = n_neurons + out_features  # inter + command + motor

        self.n_command = max(int(0.4 * n_neurons), 1)
        self.n_inter = n_neurons - self.n_command

        self.counts, self.n_connections = self._set_counts(
            in_features,
            out_features,
        )
        self.masks = self._init_masks(in_features)
        self.cmd_recurrent = torch.zeros(
            (self.counts.command, self.counts.command),
            dtype=torch.int32,
        )

        self.build()

    def config(self) -> WiringConfig:
        """Returns parameters as a config."""
        return WiringConfig(n_neurons=self.counts, **self.__dict__)

    def _init_masks(self, n_inputs: int) -> LayerMasks:
        """Create all layer masks."""
        return LayerMasks(
            inter=torch.zeros(
                (n_inputs, self.counts.inter),
                dtype=torch.int32,
            ),
            command=torch.zeros(
                (self.counts.inter, self.counts.command),
                dtype=torch.int32,
            ),
            motor=torch.zeros(
                (self.counts.command, self.counts.motor),
                dtype=torch.int32,
            ),
        )

    def _synapse_count(self, count: int, scale: int = 1) -> int:
        """
        A helper method for computing the synapse count.

        Parameters:
            count (int): the number of neurons
            scale (int, optional): a scale factor. Default is '1'
        """
        return max(int(count * self.density_level * scale), 1)

    def _set_counts(
        self, in_features: int, out_features: int
    ) -> Tuple[NeuronCounts, ConnectionCounts]:
        """Computes the node layer and connection counts."""
        counts = NeuronCounts(
            sensory=in_features,
            inter=self.n_inter,
            command=self.n_command,
            motor=out_features,
        )

        connections = ConnectionCounts(
            sensory=self._synapse_count(self.n_inter),
            inter=self._synapse_count(self.n_command),
            command=self._synapse_count(self.n_command, scale=2),
            motor=self._synapse_count(self.n_command),
        )

        return counts, connections

    @staticmethod
    def polarity(shape: tuple[int, ...] = (1,)) -> torch.IntTensor:
        """
        Randomly selects a polarity of `-1` or `1` based on shape
        and returns the results as a torch tensor.
        """
        return torch.IntTensor(np.random.choice([-1, 1], shape))

    def _build_connections(self, mask: torch.Tensor, count: int) -> torch.Tensor:
        """
        Randomly assigns connections to a set of nodes by populating its mask.

        Performs two operations:
        1. Applies minimum connections (count) to all nodes.
        2. Checks all nodes have at least 1 connection.
            If not, adds a connection to 'missing' nodes.

        Parameters:
            mask (torch.Tensor): the mask matrix.
            count (int): the number of connections per node.

        Example:
            Given 2 sensory (input) nodes and 5 inter neurons, we can define
            our first layer (inter) mask as:
            ```python
            import torch

            inter_mask = torch.zeros((2, 5), dtype=torch.int32)
            n_connections = 2

            inter_mask = build_connections(inter_mask, n_connections)

            # tensor([[-1,  1,  0,  0,  1],
            #         [ 0,  0, -1, -1,  0]], dtype=torch.int32)
            ```
        """
        num_nodes, num_cols = mask.shape

        # Add required connection count
        col_indices = torch.IntTensor(
            np.random.choice(num_cols, (num_nodes, count)),
        )
        polarities = self.polarity(col_indices.shape)
        row_indices = torch.arange(num_nodes).unsqueeze(1)

        mask[row_indices, col_indices] = polarities

        # Add missing node connections (if applicable)
        # -> Every node in 'num_cols' must have at least 1 connection
        # -> Column with all 0s = non-connected node
        is_col_all_zero = (mask == 0).all(dim=0)
        col_zero_indices = torch.nonzero(is_col_all_zero, as_tuple=True)[0]
        zero_count = col_zero_indices.numel()

        if zero_count > 0:
            # For each missing connection, randomly select a node and add connection
            # -> row = node
            row_indices = torch.randint(0, num_nodes, (zero_count,))
            random_polarities = self.polarity((zero_count,))
            mask[row_indices, col_zero_indices] = random_polarities

        return mask

    def _build_recurrent_connections(
        self, array: torch.Tensor, count: int
    ) -> torch.Tensor:
        """
        Adds recurrent connections to a set of nodes.

        Used for the command neurons to simulate bidirectional
        connections.

        Parameters:
            array (torch.Tensor): the matrix array to apply changes to
            count (int): total number of connections to add
        """
        n_nodes = array.shape[0]

        src = np.random.choice(n_nodes, count)
        dest = np.random.choice(n_nodes, count)
        polarities = self.polarity((count,))

        array[src, dest] = polarities
        return array

    def build(self) -> None:
        """
        Builds the mask connections for each layer.

        The NCP has three layers with separate masks:
        1. Sensory -> inter
        2. Inter -> command
        3. Command -> motor

        Plus, command recurrent connections for visualization.
        """
        # Sensory -> inter
        self.masks.inter = self._build_connections(
            self.masks.inter,
            self.counts.sensory,
        )
        # Inter -> command
        self.masks.command = self._build_connections(
            self.masks.command,
            self.counts.inter,
        )
        # Command -> motor
        self.masks.motor = self._build_connections(
            self.masks.motor,
            self.counts.command,
        )

        # Command -> command
        self.cmd_recurrent = self._build_recurrent_connections(
            self.cmd_recurrent,
            self.counts.command,
        )

    def data(self) -> Tuple[LayerMasks, NeuronCounts]:
        """
        Retrieves wiring storage containers for layer masks and node counts.
        """
        return self.masks, self.counts
