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

from dataclasses import fields
from typing import Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np

from velora.base.spec import HeadSpec, LayerSpec
from velora.lnn.spec import NCPWiringSpec


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
) -> jax.Array:
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


def build_ncp_wiring(
    in_features: int,
    n_neurons: int,
    head_spec_cls: Type[HeadSpec],
    *,
    seed: int = 28,
    sparsity: float = 0.5,
    **head_attrs: int,
) -> NCPWiringSpec:
    """
    Build an NCP wiring specification.

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    head_spec_cls : Type[HeadSpec]
        The `HeadSpec` class to use.
        Valid options: `[ACMHeadSpec, OCMHeadSpec, DiscoHeadSpec, SingleHeadSpec]`
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Must be a value between `[0.1, 0.9]`:

        - Where `0.1` neurons are very dense
        - Where `0.9` neurons are very sparse

        Default is `0.5`
    **head_attrs : int
        Kwargs matching the head spec class's field names.
        E.g., for `ACMHeadSpec`: `z=64, aux_pi=4, q=1`

    Returns
    -------
    wiring : NCPWiringSpec
        NCP wiring specification

    Raises
    ------
    invalid_sparsity : ValueError
        If `sparsity` is outside `[0.1, 0.9]`
    invalid_spec : TypeError
        If `head_spec_cls` is not a subclass of `HeadSpec`
    invalid_fields : ValueError
        If provided field names don't match the spec's expected fields
    """
    if not 0.1 <= sparsity <= 0.9:
        raise ValueError(f"'{sparsity=}' must be between '[0.1, 0.9]'.")

    if not (isinstance(head_spec_cls, type) and issubclass(head_spec_cls, HeadSpec)):
        raise TypeError(
            f"`head_spec_cls` must be a `HeadSpec` subclass. "
            f"Got `{type(head_spec_cls).__name__}`"
        )

    expected_fields = {f.name for f in fields(head_spec_cls)}
    provided_fields = set(head_attrs.keys())
    missing = expected_fields - provided_fields
    extra = provided_fields - expected_fields

    if missing or extra:
        raise ValueError(
            f"Unknown fields for `{head_spec_cls.__name__}`: `{sorted(missing)}`. "
            f"Expected: `{sorted(expected_fields)}`"
        )

    density = 1.0 - sparsity
    n_command = max(int(0.4 * n_neurons), 1)
    n_inter = n_neurons - n_command

    rng = np.random.default_rng(seed)

    # Motor heads: command -> each output dim
    motor_conn = _synapse_count(n_command, density, scale=2)
    head_layers = {
        name: _build_layer((n_command, dim), motor_conn, rng)
        for name, dim in head_attrs.items()
    }
    head_spec = head_spec_cls(**head_layers)

    # Core layers
    inter_count = _synapse_count(n_inter, density)
    command_count = _synapse_count(n_command, density)

    # sensory -> inter
    inter = _build_layer((in_features, n_inter), inter_count, rng)
    # inter -> command
    command = _build_layer((n_inter, n_command), command_count, rng)

    return NCPWiringSpec(inter=inter, command=command, motor=head_spec)
