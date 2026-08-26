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
from dataclasses import dataclass, fields, replace
from typing import Any, Self

import torch


@dataclass(frozen=True)
class ExperienceBatch:
    """
    A batch of experience for buffers.

    Parameters
    ----------
    obs : torch.Tensor
        The environment observations `(capacity, n_envs, *obs_shape)`
    actions : torch.Tensor
        The agent actions `(capacity, n_envs, *act_shape)`
    rewards : torch.Tensor
        The rewards received from the environments `(capacity, n_envs)`
    dones : torch.Tensor
        The environment completion flags `(capacity, n_envs)`
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    def flatten(self) -> Self:
        """
        Flattens the batch's time and environment dimensions into a
        single one, reducing each tensor from `(capacity, n_envs, *shape)`
        to `(capacity * n_envs, *shape)`.

        Useful for mini-batch sampling.

        Returns
        -------
        rollout : Self
            A new flattened batch of the same data
        """
        values: dict[str, torch.Tensor] = {
            f.name: getattr(self, f.name).flatten(0, 1) for f in fields(self)
        }
        return replace(self, **values)


class BufferBase(ABC):
    """
    Base contract for all buffer types. All buffers should
    inherit from this contract to maintain a standardized format.

    Parameters
    ----------
    capacity : int
        Maximum size of the rollout buffer
    device : torch.device
        Device to load tensors onto
    """

    def __init__(self, capacity: int, *, device: torch.device) -> None:
        self.device = device
        self.capacity = capacity

        self._position = 0
        self._size = 0

    @property
    def full(self) -> bool:
        """Check if buffer is full."""
        return self._size >= self.capacity

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Add samples to the buffer."""
        ...

    @abstractmethod
    def sample(self) -> Any:
        """
        Retrieve samples from the buffer.

        Returns
        -------
        exp : Any
            A set of experience
        """
        ...

    def __len__(self) -> int:
        """Current size of the buffer."""
        return self._position
