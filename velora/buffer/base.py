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


from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, Self

import torch


@dataclass(frozen=True)
class BufferSamples:
    """
    A batch of experience for buffers.

    Parameters
    ----------
    obs : torch.Tensor
        The environment observations `(capacity, num_envs, *obs_shape)`
    actions : torch.Tensor
        The agent actions `(capacity, num_envs, *act_shape)`
    rewards : torch.Tensor
        The rewards received from the environments `(capacity, num_envs, 1)`
    dones : torch.Tensor
        The environment completion flags `(capacity, num_envs, 1)`
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        """
        Apply a function to every tensor in the batch.

        Parameters
        ----------
        fn : Callable[[torch.Tensor], torch.Tensor]
            The function to apply to each tensor

        Returns
        -------
        batch : Self
            A new batch containing the transformed tensors
        """
        values: dict[str, torch.Tensor] = {
            f.name: fn(getattr(self, f.name)) for f in fields(self)
        }
        return replace(self, **values)

    def flatten(self) -> Self:
        """
        Flattens the batch's time and environment dimensions into a
        single one, reducing each tensor from `(capacity, num_envs, *shape)`
        to `(capacity * num_envs, *shape)`.

        Returns
        -------
        rollout : Self
            A new flattened batch of the same data
        """
        return self.map(lambda t: t.flatten(0, 1))

    def select(self, indices: torch.Tensor, *, dim: int = 0) -> Self:
        """
        Selects the same indices from every tensor in the batch along
        a given dimension.

        Useful for mini-batch sampling after `flatten()`.

        Parameters
        ----------
        indices : torch.Tensor
            A 1D tensor of indices to select
        dim : int (optional)
            The dimension to select along. Default is `0`

        Returns
        -------
        batch : Self
            A new batch containing only the selected entries
        """
        return self.map(lambda t: t.index_select(dim, indices))


class RLBuffer(ABC):
    """
    An abstract base class for all Reinforcement Learning buffers.

    Buffers are tensors with shape `(capacity, num_envs, *data_shape)`.

    Parameters
    ----------
    capacity : int
        Maximum storage size of the buffer
    device : torch.device
        Device to load tensors onto
    num_envs : int (optional)
        Number of parallel environments used. Default is `1`
    """

    def __init__(
        self,
        capacity: int,
        *,
        device: torch.device,
        num_envs: int = 1,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.num_envs = num_envs

        self._position = 0
        self._size = 0

    @property
    def full(self) -> bool:
        """Check if buffer is full."""
        return self._size >= self.capacity

    def __len__(self) -> int:
        """Current size of the buffer."""
        return self._size

    def _advance(self) -> None:
        """
        Move the write position forward by one timestep,
        wrapping around at capacity.
        """
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the buffers memory by clearing internal flags
        and data.
        """
        self._position = 0
        self._size = 0

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Add samples to the buffer."""
        ...

    @abstractmethod
    def sample(self, *args, **kwargs) -> Any:
        """
        Retrieve samples from the buffer.

        Returns
        -------
        samples : Any
            Samples of data extracted from the buffer
        """
        ...
