try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # pragma: no cover

import torch

from velora.buffer.base import BufferBase
from velora.buffer.experience import BatchExperience
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
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        if len(self) == self.capacity:
            raise BufferError("Buffer full! Use the 'empty()' method first.")

        super().add(state, action, reward, next_state, done)

    @override
    def sample(self) -> BatchExperience:
        """
        Returns the entire rollout buffer as a batch of experience.

        Returns:
            batch (BatchExperience): an object of samples with the attributes (`states`, `actions`, `rewards`, `next_states`, `dones`).

                All items have the same shape `(batch_size, features)`.
        """
        if len(self) == 0:
            raise BufferError("Buffer is empty!")

        return BatchExperience(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            dones=self.dones,
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
