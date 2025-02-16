from collections import deque
from dataclasses import dataclass, astuple
import random
from typing import List, Tuple

import torch

from velora.models.utils import to_tensor


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
        device (torch.device, optional): the PyTorch device to load tensors onto.
            Default is `None`
    """

    def __init__(self, capacity: int, *, device: torch.device | None = None) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, exp: Experience) -> None:
        """Stores an experience in the buffer."""
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> BatchExperience:
        """Returns a random batch of experiences."""
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer does not contain enough experiences. Available: {len(self.buffer)}, Requested: {batch_size}"
            )

        batch: List[Experience] = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return BatchExperience(
            states=to_tensor(states, stack=True, device=self.device),
            actions=to_tensor(actions, device=self.device),
            rewards=to_tensor(rewards, device=self.device, unsqueeze=1),
            next_states=to_tensor(next_states, stack=True, device=self.device),
            dones=to_tensor(dones, device=self.device, unsqueeze=1),
        )

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.buffer)


class RolloutBuffer:
    """
    A Rollout Buffer for storing agent experiences. Used for On-Policy agents.

    Parameters:
        capacity (int): Maximum rollout length
        device (torch.device, optional): the PyTorch device to load tensors onto.
            Default is `None`
    """

    def __init__(self, capacity: int, *, device: torch.device | None = None) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

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
            states=to_tensor(states, stack=True, device=self.device, unsqueeze=1),
            actions=to_tensor(actions, device=self.device),
            rewards=to_tensor(rewards, device=self.device, unsqueeze=1),
            next_states=to_tensor(
                next_states,
                stack=True,
                device=self.device,
                unsqueeze=1,
            ),
            dones=to_tensor(dones, device=self.device, unsqueeze=1),
        )

    def clear(self) -> None:
        """Empties the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.buffer)
