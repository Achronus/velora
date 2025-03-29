try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # pragma: no cover

from typing import Generator

import torch

from velora.buffer.base import BufferBase
from velora.buffer.experience import RolloutBatchExperience
from velora.models.config import BufferConfig


class RolloutBuffer(BufferBase):
    """
    A Rollout Buffer for storing agent experiences. Used for On-Policy agents.

    Uses a similar implementation to `ReplayBuffer`. However, it must
    be emptied after it is full.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            capacity (int): Maximum rollout length
            state_dim (int): dimension of state observations
            action_dim (int): dimension of actions
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(capacity, state_dim, action_dim, device=device)

        self.log_probs = torch.zeros((capacity, 1), device=device)
        self.values = torch.zeros((capacity, 1), device=device)

    def config(self) -> BufferConfig:
        """
        Creates a buffer config model.

        Returns:
            config (BufferConfig): a config model with buffer details.
        """
        return BufferConfig(
            type="RolloutBuffer",
            capacity=self.capacity,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    @override
    def add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Adds a set of experience to the buffer. Useful for vectorized environments.

        Parameters:
            states (torch.Tensor): current state observations
            actions (torch.Tensor): actions taken
            rewards (torch.Tensor): rewards received
            next_states (torch.Tensor): next state observations
            dones (torch.Tensor): episode completions
            log_probs (torch.Tensor): log probabilities
            values (torch.Tensor): state values
        """
        if len(self) == self.capacity:
            raise BufferError("Buffer full! Use the 'empty()' method first.")

        n_items = states.shape[0]
        end_idx = self.position + n_items

        self.states[self.position : end_idx] = states.to(torch.float32)
        self.actions[self.position : end_idx] = actions
        self.rewards[self.position : end_idx] = rewards.unsqueeze(-1)
        self.next_states[self.position : end_idx] = next_states.to(torch.float32)
        self.dones[self.position : end_idx] = dones.unsqueeze(-1)
        self.log_probs[self.position : end_idx] = log_probs
        self.values[self.position : end_idx] = values

        # Update position - deque style
        self.position = (self.position + n_items) % self.capacity
        self.size = min(self.size + n_items, self.capacity)

    @override
    def sample(self, batch_size: int) -> Generator[RolloutBatchExperience, None, None]:
        """
        Returns a generator that yields mini-batches of experience.
        Randomly shuffles samples first.

        Parameters:
            batch_size (int): number of samples per mini-batch

        Yields:
            mini_batch (RolloutBatchExperience): an object of samples with the attributes (`states`, `actions`, `rewards`, `next_states`, `dones`, `log_probs`, `values`).

                All items have the same shape `(batch_size, features)`.
        """
        if len(self) == 0:
            raise BufferError("Buffer is empty!")

        if len(self) != self.capacity:
            raise BufferError(
                f"Buffer must be filled first! ({len(self)}/{self.capacity})"
            )

        indices = torch.randperm(self.capacity, device=self.device)
        n_batches = self.capacity // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size if i < n_batches - 1 else self.capacity
            batch_indices = indices[start_idx:end_idx]

            # Create and yield one batch at a time
            yield RolloutBatchExperience(
                states=self.states[batch_indices],
                actions=self.actions[batch_indices],
                rewards=self.rewards[batch_indices],
                next_states=self.next_states[batch_indices],
                dones=self.dones[batch_indices],
                log_probs=self.log_probs[batch_indices],
                values=self.values[batch_indices],
            )

    def empty(self) -> None:
        """Empties the buffer."""
        self.position = 0
        self.size = 0

        # Reset tensors
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        self.dones.zero_()
        self.log_probs.zero_()
        self.values.zero_()

    def is_full(self) -> bool:
        """Checks if the buffer is full."""
        return len(self) == self.capacity
