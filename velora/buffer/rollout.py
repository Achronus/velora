from typing import override

import torch

from velora.buffer.base import BufferBase
from velora.buffer.experience import BatchExperience, Experience
from velora.models.config import BufferConfig


class RolloutBuffer(BufferBase):
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
        super().__init__(capacity, device=device)

    @property
    def config(self) -> BufferConfig:
        """
        Creates a buffer config model.

        Returns:
            config (BufferConfig): a config model with buffer details.
        """
        return BufferConfig(type="RolloutBuffer", capacity=self.capacity)

    @override
    def push(self, exp: Experience) -> None:
        """
        Stores an experience in the buffer.

        Parameters:
            exp (Experience): a single set of experience as an object
        """
        if len(self.buffer) == self.capacity:
            raise BufferError("Buffer full! Use the 'empty()' method first.")

        super().push(exp)

    @override
    def sample(self) -> BatchExperience:
        """
        Returns the entire rollout buffer as a batch of experience.

        Returns:
            batch (BatchExperience): an object of samples with the attributes (`states`, `actions`, `rewards`, `next_states`, `dones`).

                All items have the same shape `(batch_size, features)`.
        """
        if len(self.buffer) == 0:
            raise BufferError("Buffer is empty!")

        return self._batch(self.buffer)

    def empty(self) -> None:
        """Empties the buffer."""
        self.buffer.clear()
