from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Self, Tuple

import gymnasium as gym
import torch

from velora.models.config import RLAgentConfig, TrainConfig


class RLAgent:
    """
    A base class for RL agents.

    Provides a blueprint describing the core methods that agents *must* have and
    includes useful utility methods.
    """

    def __init__(self, device: torch.device | None) -> None:
        """
        Parameters:
            device (torch.device, optional): the device to perform computations on
        """
        self.device = device
        self.config: RLAgentConfig | None = None

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

    def _set_train_params(self, params: Dict[str, Any]) -> TrainConfig:
        """
        Helper method. Sets the `train_params` given a dictionary of training parameters.

        Parameters:
            params (Dict[str, Any]): a dictionary of training parameters

        Returns:
            config (TrainConfig): a training config model
        """
        return TrainConfig(
            callbacks=(
                dict(cb.config() for cb in params["callbacks"])
                if params["callbacks"]
                else None
            ),
            **{
                k: v for k, v in params.items() if k not in ["self", "env", "callbacks"]
            },
        )
