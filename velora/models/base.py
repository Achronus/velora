from abc import abstractmethod
from pathlib import Path
from typing import Any, Self, Tuple

import gymnasium as gym
import torch


class RLAgent:
    """
    An abstract base class for RL agents.

    Provides a blueprint describing the core methods that agents *must* have.
    """

    def __init__(self, device: torch.device | None) -> None:
        """
        Parameters:
            device (torch.device, optional): the device to perform computations on
        """
        self.device = device

    @abstractmethod
    def train(
        self,
        env: gym.Env,
        batch_size: int,
        n_episodes: int,
        max_steps: int,
        window_size: int,
        *args,
        **kwargs,
    ) -> Any:
        pass  # pragma: no cover

    @abstractmethod
    def predict(
        self, state: torch.Tensor, hidden: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass  # pragma: no cover

    @abstractmethod
    def save(self, filepath: str | Path, *, buffer: bool = False) -> None:
        pass  # pragma: no cover

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path, *, buffer: bool = False) -> Self:
        pass  # pragma: no cover
