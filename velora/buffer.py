from collections import deque
from dataclasses import dataclass, astuple
import random
from typing import Any, List, Tuple

import torch


@dataclass
class Experience:
    """A single agent experience."""

    state: torch.Tensor
    action: float
    reward: float
    next_state: torch.Tensor
    done: bool

    def __iter__(self) -> Tuple:
        """Unpack experience instances as tuples."""
        return iter(astuple(self))


@dataclass
class BatchExperience:
    """A batch of agent experiences."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """
    A Buffer for storing agent experiences. Used for Off-Policy agents.

    Parameters:
        capacity (int): the total capacity of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    @staticmethod
    def _to_tensor(
        item: List[Any],
        stack: bool = False,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        A helper method to convert an item to a tensor.

        Parameters:
            item (List[Any]): a list of items of any type
            stack (bool, optional): whether to stack the values into a single
                tensor. Default is 'False'
            dtype (torch.dtype, optional): the data type for the tensor.
                Default is 'torch.float32'
        """
        if isinstance(item, torch.Tensor):
            return item

        if stack:
            return torch.stack(item).to(dtype)

        return torch.tensor(item, dtype=dtype)

    def push(self, exp: Experience) -> None:
        """Stores an experience in the buffer."""
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> Tuple:
        """Returns a random batch of experiences."""
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer does not contain enough experiences. Available: {len(self.buffer)}, Requested: {batch_size}"
            )

        batch: List[Experience] = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return BatchExperience(
            states=self._to_tensor(states, stack=True),
            actions=self._to_tensor(actions),
            rewards=self._to_tensor(rewards),
            next_states=self._to_tensor(next_states, stack=True),
            dones=self._to_tensor(dones),
        )

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.buffer)


class RolloutBuffer:
    """
    A Rollout Buffer for storing agent experiences. Used for On-Policy agents.

    Parameters:
        capacity (int): Maximum rollout length
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    @staticmethod
    def _to_tensor(
        item: List[Any],
        stack: bool = False,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        A helper method to convert an item to a tensor.

        Parameters:
            item (List[Any]): a list of items of any type
            stack (bool, optional): whether to stack the values into a single
                tensor. Default is 'False'
            dtype (torch.dtype, optional): the data type for the tensor.
                Default is 'torch.float32'
        """
        if isinstance(item, torch.Tensor):
            return item

        if stack:
            return torch.stack(item).to(dtype)

        return torch.tensor(item, dtype=dtype)

    def push(self, exp: Experience) -> None:
        """Stores an experience in the buffer."""
        if len(self.buffer) == self.capacity:
            raise BufferError("Buffer full! Use the 'clear()' method first.")

        self.buffer.append(exp)

    def sample(self) -> BatchExperience:
        """Returns the entire rollout buffer as a batch."""
        if len(self.buffer) == 0:
            raise BufferError("Buffer is empty!")

        states, actions, rewards, next_states, dones = zip(*self.buffer)

        return BatchExperience(
            states=self._to_tensor(states, stack=True),
            actions=self._to_tensor(actions),
            rewards=self._to_tensor(rewards),
            next_states=self._to_tensor(next_states, stack=True),
            dones=self._to_tensor(dones),
        )

    def clear(self) -> None:
        """Empties the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.buffer)
