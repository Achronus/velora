from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class NeuronCounts:
    """
    Storage container for NCP neuron category counts.

    Parameters:
        sensory (int): number of input nodes
        inter (int): number of decision nodes
        command (int): number of high-level decision nodes
        motor (int): number of output nodes
    """

    sensory: int
    inter: int
    command: int
    motor: int


@dataclass
class SynapseCounts:
    """
    Storage container for NCP neuron synapse connection counts.

    Parameters:
        sensory (int): number of connections for input nodes
        inter (int): number of connections for decision nodes
        command (int): number of connections for high-level decision nodes
        motor (int): number of connections for output nodes
    """

    sensory: int
    inter: int
    command: int
    motor: int


@dataclass
class LayerMasks:
    """
    Storage container for layer masks.

    Parameters:
        inter (torch.Tensor): sparse weight mask for input layer
        command (torch.Tensor): sparse weight mask for hidden layer
        motor (torch.Tensor): sparse weight mask for output layer
        recurrent (torch.Tensor): sparse weight mask for recurrent connections
    """

    inter: torch.Tensor
    command: torch.Tensor
    motor: torch.Tensor
    recurrent: torch.Tensor


class Wiring:
    """
    Creates sparse wiring masks for Neural Circuit Policy (NCP) Networks.

    !!! note

        NCPs have three layers:

        1. Inter (input)
        2. Command (hidden)
        3. Motor (output)
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        sparsity_level: float = 0.5,
    ) -> None:
        """
        Parameters:
            in_features (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes)
            out_features (int): number of outputs (motor nodes)
            sparsity_level (float, optional): controls the connection sparsity between neurons.

                Must be a value between `[0.1, 0.9]` -

                - When `0.1` neurons are very dense.
                - When `0.9` neurons are very sparse.
        """
        if sparsity_level < 0.1 or sparsity_level > 0.9:
            raise ValueError(f"'{sparsity_level=}' must be between '[0.1, 0.9]'.")

        self.density_level = 1.0 - sparsity_level

        self.n_command = max(int(0.4 * n_neurons), 1)
        self.n_inter = n_neurons - self.n_command

        self.counts, self._n_connections = self._set_counts(
            in_features,
            out_features,
        )
        self.masks = self._init_masks(in_features)

        self.build()

    @property
    def n_connections(self) -> SynapseCounts:
        """
        Neuron connection counts.

        Returns:
            connections (SynapseCounts): object containing neuron connection counts.
        """
        return self._n_connections

    def _init_masks(self, n_inputs: int) -> LayerMasks:
        """
        Helper method. Initializes all layer masks with zeros
        and stores them in a container.

        Parameters:
            n_inputs (int): the number of input nodes in the layer

        Returns:
            masks (LayerMasks): initialized layer masks.
        """
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
            recurrent=torch.zeros(
                (self.counts.command, self.counts.command),
                dtype=torch.int32,
            ),
        )

    def _synapse_count(self, count: int, scale: int = 1) -> int:
        """
        Helper method. Computes the synapse count for a single layer.

        Parameters:
            count (int): the number of neurons
            scale (int, optional): a scale factor

        Returns:
            count (int): synapse count.
        """
        return max(int(count * self.density_level * scale), 1)

    def _set_counts(
        self, in_features: int, out_features: int
    ) -> Tuple[NeuronCounts, SynapseCounts]:
        """
        Helper method. Computes the node layer and connection counts.

        Parameters:
            in_features (int): number of network input nodes
            out_features (int): number of network output nodes

        Returns:
            neuron_counts (NeuronCounts): object with neuron counts.
            synapse_counts (SynapseCounts): object with synapse connection counts.
        """
        counts = NeuronCounts(
            sensory=in_features,
            inter=self.n_inter,
            command=self.n_command,
            motor=out_features,
        )

        connections = SynapseCounts(
            sensory=self._synapse_count(self.n_inter),
            inter=self._synapse_count(self.n_command),
            command=self._synapse_count(self.n_command, scale=2),
            motor=self._synapse_count(self.n_command),
        )

        return counts, connections

    @staticmethod
    def polarity(shape: Tuple[int, ...] = (1,)) -> torch.IntTensor:
        """
        Utility method. Randomly selects a polarity of `-1` or `1`, `n` times
        based on shape.

        Parameters:
            shape (Tuple[int, ...]): size of the polarity matrix to generate.

        Returns:
            matrix (torch.Tensor): a polarity matrix filled with `-1` and `1`.
        """
        return torch.IntTensor(np.random.choice([-1, 1], shape))

    def _build_connections(self, mask: torch.Tensor, count: int) -> torch.Tensor:
        """
        Helper method. Randomly assigns connections to a set of nodes by populating
        its mask.

        !!! note "Performs two operations"

            1. Applies minimum connections (count) to all nodes.
            2. Checks all nodes have at least 1 connection.
                If not, adds a connection to 'missing' nodes.

        Parameters:
            mask (torch.Tensor): the initialized mask
            count (int): the number of connections per node

        Examples:
            Given 2 sensory (input) nodes and 5 inter neurons, we can define
            our first layer (inter) mask as:

            ```python
            import torch

            inter_mask = torch.zeros((2, 5), dtype=torch.int32)
            n_connections = 2

            inter_mask = wiring._build_connections(inter_mask, n_connections)

            # tensor([[-1,  1,  0,  0,  1],
            #         [ 0,  0, -1, -1,  0]], dtype=torch.int32)
            ```

        Returns:
            mask (torch.Tensor): updated layer sparsity mask.
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
        Utility method. Adds recurrent connections to a set of nodes.

        Used to simulate bidirectional connections between command neurons. Strictly
        used for visualization purposes.

        Parameters:
            array (torch.Tensor): an initialized matrix to update
            count (int): total number of connections to add

        Returns:
            matrix (torch.Tensor): the updated matrix.
        """
        n_nodes = array.shape[0]

        src = np.random.choice(n_nodes, count)
        dest = np.random.choice(n_nodes, count)
        polarities = self.polarity((count,))

        array[src, dest] = polarities
        return array

    def build(self) -> None:
        """
        Builds the mask wiring for each layer.

        !!! note "Layer format"

            Follows a three layer format, each with separate masks:

            1. Sensory -> inter
            2. Inter -> command
            3. Command -> motor

        Plus, command recurrent connections for ODE solvers.
        """
        # Sensory -> inter
        self.masks.inter = self._build_connections(
            self.masks.inter,
            self._n_connections.sensory,
        )
        # Inter -> command
        self.masks.command = self._build_connections(
            self.masks.command,
            self._n_connections.inter,
        )
        # Command -> motor
        self.masks.motor = self._build_connections(
            self.masks.motor,
            self._n_connections.command,
        )

        # Command -> command
        self.masks.recurrent = self._build_recurrent_connections(
            self.masks.recurrent,
            self._n_connections.command,
        )

    def data(self) -> Tuple[LayerMasks, NeuronCounts]:
        """
        Retrieves wiring storage containers for layer masks and node counts.

        Returns:
            masks (LayerMasks): the object containing layer masks.
            counts (NeuronCounts): the object containing node counts.
        """
        return self.masks, self.counts
