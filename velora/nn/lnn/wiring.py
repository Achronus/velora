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

from typing import NamedTuple

import numpy as np
import torch


class Wiring(NamedTuple):
    """
    NCP wiring masks for a CfC-LNN network.

    All masks use the PyTorch layout `(out_features, in_features)` with
    binary connection values `(0, 1)`. Layer sizes are derivable from
    mask shapes, e.g. `inter.shape[0]` is the number of inter neurons.

    Parameters
    ----------
    inter : torch.Tensor
        Inter (input) layer mask of shape `(n_inter, in_features)`
    command : torch.Tensor
        Command (hidden) layer mask of shape `(n_command, n_inter)`
    recurrent : torch.Tensor
        Command layer hidden-to-hidden mask of shape
        `(n_command, n_command)`
    heads : dict[str, torch.Tensor]
        Motor (output) head masks keyed by head name, each of shape
        `(head_dim, n_command)`. Order matches the `heads` argument
        given to `build_wiring`
    """

    inter: torch.Tensor
    command: torch.Tensor
    recurrent: torch.Tensor
    heads: dict[str, torch.Tensor]


def _check_sparsity(sparsity: float, name: str) -> None:
    """
    Utility method that validates a sparsity level.

    Parameters
    ----------
    sparsity : float
        The sparsity level to validate
    name : str
        The argument name, used in the error message

    Raises
    ------
    invalid_sparsity : ValueError
        When `sparsity` is outside `[0.1, 0.9]`
    """
    if not 0.1 <= sparsity <= 0.9:
        raise ValueError(f"'{name}={sparsity}' must be between '[0.1, 0.9]'.")


def _binomial_choice(
    n: int,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Utility method that draws a random set of distinct neuron indices.

    The number of indices is drawn from `Binomial(n, density)`, clipped
    to at least `1`, matching the distribution used to populate the mask
    itself.

    Parameters
    ----------
    n : int
        The number of neurons to choose from
    density : float
        The density of the layer connections (`1.0 - sparsity`)
    rng : np.random.Generator
        NumPy random number generator

    Returns
    -------
    indices : np.ndarray
        Distinct neuron indices
    """
    count = int(np.clip(rng.binomial(n, density), 1, n))
    return rng.choice(n, count, replace=False)


def _make_mask(
    shape: tuple[int, int],
    density: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Randomly assigns connections between two layers of neurons.

    Each potential synapse is drawn independently from `Bernoulli(density)`,
    making the number of connections per neuron `Binomial(n, density)` and
    their targets uniform.

    Note -
        Performs two operations:

        1. Samples the connections between every source and target pair
        2. Ensures no neuron is isolated, giving any target without
           incoming connections, and any source without outgoing
           connections, a random set of both

    Parameters
    ----------
    shape : tuple[int, int]
        Mask shape `(n_sources, n_targets)`
    density : float
        The density of the layer connections (`1.0 - sparsity`)
    rng : np.random.Generator
        NumPy random number generator

    Returns
    -------
    mask : torch.Tensor
        Populated connection mask, transposed to the PyTorch layout
        `(n_targets, n_sources)`
    """
    n_sources, n_targets = shape
    mask = rng.random(shape) < density

    for target in np.flatnonzero(~mask.any(axis=0)):
        mask[_binomial_choice(n_sources, density, rng), target] = True

    for source in np.flatnonzero(~mask.any(axis=1)):
        mask[source, _binomial_choice(n_targets, density, rng)] = True

    return torch.asarray(mask, dtype=torch.float32).T.contiguous()


def _make_recurrent_mask(
    n_neurons: int,
    sparsity: float | None,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Utility method that builds a layer's hidden-to-hidden mask.

    Self-loops are permitted and neurons are not repaired, since an
    isolated neuron still receives incoming connections from the
    previous layer.

    Parameters
    ----------
    n_neurons : int
        The number of neurons in the layer
    sparsity : float | None
        Controls the recurrent connection sparsity. When `None`, the
        mask is dense
    rng : np.random.Generator
        NumPy random number generator

    Returns
    -------
    mask : torch.Tensor
        The recurrent mask of shape `(n_neurons, n_neurons)`
    """
    if sparsity is None:
        return torch.ones((n_neurons, n_neurons))

    mask = rng.random((n_neurons, n_neurons)) < (1.0 - sparsity)
    return torch.asarray(mask, dtype=torch.float32)


def build_layer_mask(
    in_features: int,
    out_features: int,
    *,
    seed: int = 28,
    sparsity: float = 0.5,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Build an NCP connection mask for a single layer.

    A standalone variant of `build_wiring` for wiring one layer in
    isolation.

    Parameters
    ----------
    in_features : int
        Number of input nodes
    out_features : int
        Number of output nodes
    seed : int (optional)
        Random number generator seed. Default is `28`
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Must be a value between `[0.1, 0.9]`:

        - Where `0.1` neurons are very dense
        - Where `0.9` neurons are very sparse

        Default is `0.5`
    rng : np.random.Generator (optional)
        An existing NumPy random number generator to use instead of
        creating one from `seed`. Useful for wiring multiple layers
        from a single generator. Default is `None`

    Returns
    -------
    mask : torch.Tensor
        The populated connection mask with binary values `(0, 1)`,
        in the PyTorch layout `(out_features, in_features)`

    Raises
    ------
    invalid_sparsity : ValueError
        When `sparsity` is outside `[0.1, 0.9]`

    Examples
    --------
    ```python
    mask = build_layer_mask(4, 16)
    mask.shape  # (16, 4)
    ```
    """
    _check_sparsity(sparsity, "sparsity")

    rng = np.random.default_rng(seed) if rng is None else rng
    return _make_mask((in_features, out_features), 1.0 - sparsity, rng)


def build_wiring(
    in_features: int,
    n_neurons: int,
    heads: dict[str, int],
    *,
    seed: int = 28,
    sparsity: float = 0.5,
    recurrent_sparsity: float | None = None,
) -> Wiring:
    """
    Build the connection masks for a CfC-LNN NCP network.

    Splits `n_neurons` into inter and command nodes using:

    ```python
    n_command = max(int(0.4 * n_neurons), 1)
    n_inter = n_neurons - n_command
    ```

    Then wires three layer groups in sequence:

    1. Inter (input) - `sensory -> inter`
    2. Command (hidden) - `inter -> command`
    3. Motor (output) - `command -> head`, one per entry in `heads`

    Plus a hidden-to-hidden mask for the command layer.

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter + command nodes)
    heads : dict[str, int]
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
    recurrent_sparsity : float | None (optional)
        Controls the sparsity of the command layer's hidden-to-hidden
        connections. When `None`, they are dense.
        Otherwise, must be a value between `[0.1, 0.9]`.

        The NCP specification wires these sparsely, at an equivalent
        density of `2 * (1.0 - sparsity) / n_command`. This is far
        sparser than dense (2-4% at typical widths) and removes most of
        the network's temporal memory, so it is opt-in.

        Default is `None`

    Returns
    -------
    wiring : Wiring
        The populated NCP wiring masks

    Raises
    ------
    invalid_sparsity : ValueError
        When `sparsity` or `recurrent_sparsity` is outside `[0.1, 0.9]`
    heads_missing : ValueError
        When `heads` is empty

    Examples
    --------
    ```python
    wiring = build_wiring(4, 16, {"out": 2})
    wiring.inter.shape  # (10, 4)
    wiring.command.shape  # (6, 10)
    wiring.recurrent.shape  # (6, 6)
    wiring.heads["out"].shape  # (2, 6)
    ```
    """
    _check_sparsity(sparsity, "sparsity")

    if recurrent_sparsity is not None:
        _check_sparsity(recurrent_sparsity, "recurrent_sparsity")

    if not heads:
        raise ValueError("'heads' must contain at least one entry.")

    density = 1.0 - sparsity
    n_command = max(int(0.4 * n_neurons), 1)
    n_inter = n_neurons - n_command
    rng = np.random.default_rng(seed)

    inter = build_layer_mask(in_features, n_inter, sparsity=sparsity, rng=rng)
    command = build_layer_mask(n_inter, n_command, sparsity=sparsity, rng=rng)
    recurrent = _make_recurrent_mask(n_command, recurrent_sparsity, rng)

    head_masks = {
        name: _make_mask((n_command, dim), density, rng) for name, dim in heads.items()
    }

    return Wiring(
        inter=inter,
        command=command,
        recurrent=recurrent,
        heads=head_masks,
    )
