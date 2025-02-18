from collections import deque
from dataclasses import dataclass, astuple
import random
from typing import List, Tuple

import torch

from velora.utils.torch import to_tensor, stack_tensor


@dataclass
class Experience:
    """
    Storage container for a single agent experience.

    Parameters:
        state (torch.Tensor): an environment observation
        action (float): agent action taken in the state
        reward (float): reward obtained for taking the action
        next_state (torch.Tensor): a newly generated environment observation
            after performing the action
        done (bool): environment completion status
    """

    state: torch.Tensor
    action: float
    reward: float
    next_state: torch.Tensor
    done: bool

    def __iter__(self) -> Tuple:
        """
        Iteratively unpacks experience instances as tuples.

        Best used with the [`zip()`](https://docs.python.org/3/library/functions.html#zip) method:

        ```python
        batch = [Experience(...), Experience(...), Experience(...)]
        states, actions, rewards, next_states, dones = zip(*batch)

        # ((s1, s2, s3), (a1, a2, a3), (r1, r2, r3), (ns1, ns2, ns3), (d1, d2, d3))
        ```

        Returns:
            exp (Tuple): the experience as a tuple in the form `(state, action, reward, next_state, done)`.
        """
        return iter(astuple(self))


@dataclass
class BatchExperience:
    """
    Storage container for a batch agent experiences.

    Parameters:
        states (torch.Tensor): a batch of environment observations
        actions (torch.Tensor): a batch of agent actions taken in the states
        rewards (torch.Tensor): a batch of rewards obtained for taking the actions
        next_states (torch.Tensor): a batch of newly generated environment
            observations following the actions taken
        dones (torch.Tensor): a batch of environment completion statuses
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """
    A Buffer for storing agent experiences. Used for Off-Policy agents.

    First introduced in Deep RL in the Deep Q-Network paper:
    [Player Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602).
    """

    def __init__(self, capacity: int, *, device: torch.device | None = None) -> None:
        """
        Parameters:
            capacity (int): the total capacity of the buffer
            device (torch.device, optional): the device to perform computations on
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, exp: Experience) -> None:
        """
        Stores an experience in the buffer.

        Parameters:
            exp (Experience): a single set of experience as an object
        """
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> BatchExperience:
        """
        Samples a random batch of experiences from the buffer.

        Parameters:
            batch_size (int): the number of items to sample

        Returns:
            batch (BatchExperience): an object of samples with the attributes (`states`, `actions`, `rewards`, `next_states`, `dones`).

                All items have the same shape `(batch_size, features)`.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer does not contain enough experiences. Available: {len(self.buffer)}, Requested: {batch_size}"
            )

        batch: List[Experience] = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return BatchExperience(
            states=stack_tensor(states, device=self.device),
            actions=to_tensor(actions, device=self.device).unsqueeze(1),
            rewards=to_tensor(rewards, device=self.device).unsqueeze(1),
            next_states=stack_tensor(next_states, device=self.device),
            dones=to_tensor(dones, device=self.device).unsqueeze(1),
        )

    def __len__(self) -> int:
        """
        Gets the current size of the buffer.

        Returns:
            size (int): the current size of the buffer.
        """
        return len(self.buffer)


class RolloutBuffer:
    """
    A Rollout Buffer for storing agent experiences. Used for On-Policy agents.

    Uses a similar implementation to `ReplayBuffer`. However, it must
    be emptied after it is full.
    """

    def __init__(self, capacity: int, *, device: torch.device | None = None) -> None:
        """
        Parameters:
            capacity (int): Maximum rollout length
            device (torch.device, optional): the device to perform computations on
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, exp: Experience) -> None:
        """
        Stores an experience in the buffer.

        Parameters:
            exp (Experience): a single set of experience as an object
        """
        if len(self.buffer) == self.capacity:
            raise BufferError("Buffer full! Use the 'empty()' method first.")

        self.buffer.append(exp)

    def sample(self) -> BatchExperience:
        """
        Returns the entire rollout buffer as a batch of experience.

        Returns:
            batch (BatchExperience): an object of samples with the attributes (`states`, `actions`, `rewards`, `next_states`, `dones`).

                All items have the same shape `(batch_size, features)`.
        """
        if len(self.buffer) == 0:
            raise BufferError("Buffer is empty!")

        states, actions, rewards, next_states, dones = zip(*self.buffer)

        return BatchExperience(
            states=stack_tensor(states, device=self.device),
            actions=to_tensor(actions, device=self.device).unsqueeze(1),
            rewards=to_tensor(rewards, device=self.device).unsqueeze(1),
            next_states=stack_tensor(next_states, device=self.device),
            dones=to_tensor(dones, device=self.device).unsqueeze(1),
        )

    def empty(self) -> None:
        """Empties the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """
        Gets the current size of the buffer.

        Returns:
            size (int): the current size of the buffer.
        """
        return len(self.buffer)
