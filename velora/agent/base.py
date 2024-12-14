from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from velora.analytics.base import NullAnalytics
from velora.analytics.wandb import WeightsAndBiases
from velora.config import load_config
from velora.env.gym import GymEnv
from velora.policy.base import PyTorchModel
from velora.utils import set_device

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


class AgentBase(ABC):
    """
    A base for all RL agents.

    Args:
        config_filepath (pathlib.Path | str): a YAML config filepath
        model (velora.policy.PyTorchModel): a model containing PyTorch model settings
        device (torch.device | str, optional): device to run computations on, such as `cpu`, `cuda`. When `auto` configures the device automatically (Default is `auto`)
        logging (bool, optional): a flag to disable analytic logging. If True creates a [Weights and Bias](https://wandb.ai/) instance (Default is `True`)
    """

    def __init__(
        self,
        config_filepath: Path | str,
        model: PyTorchModel,
        device: torch.device | str,
        logging: bool,
    ) -> None:
        self.config = load_config(config_filepath)
        self.device = self.assign_device(device)

        self.set_seed(self.config.run.seed)

        self.gym = GymEnv(
            config=self.config.env,
            gamma=self.config.agent.gamma,
            device=self.device,
        )
        self.analytics = NullAnalytics() if not logging else WeightsAndBiases()

        self.policy = model.policy(inputs=model.inputs).to(self.device)
        self.optimizer = model.optimizer(
            self.policy.parameters(),
            **self.config.optimizer.model_dump(),
        )
        self.loss = model.loss(**self.config.loss.model_dump())

    @property
    def env(self) -> gym.vector.SyncVectorEnv:
        """The vectorized Gymnasium environment."""
        return self.gym.env

    @staticmethod
    def assign_device(device: torch.device | str) -> torch.device:
        """Assigns the PyTorch device."""
        if device == "auto":
            return set_device()

        if isinstance(device, str):
            device = torch.device(device)

        return device

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO).
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @staticmethod
    def set_seed(seed: int) -> None:
        """Sets the seed to ensure reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)

    @abstractmethod
    def predict(self, obs: torch.Tensor) -> Any:
        """Makes a prediction for the next action based on a given observation."""
        pass  # pragma: no cover

    @abstractmethod
    def train(self) -> None:
        """Trains the agent."""
        pass  # pragma: no cover
