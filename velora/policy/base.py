from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss

from velora.policy.inputs import ExtractorInputs


class AgentModel(nn.Module, ABC):
    """
    A base class for Velora Agent models.
    """

    @abstractmethod
    def act(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Gets an action from its policy."""
        pass  # pragma: no cover

    @abstractmethod
    def get_vf(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the value function prediction."""
        pass  # pragma: no cover

    @abstractmethod
    def step(self, state: Any, next_state: Any, action: int, reward: float) -> float:
        """Performs an agent step through the environment, such as performing policy updates and setting the next action (if applicable)."""
        pass  # pragma: no cover

    @abstractmethod
    def termination(self) -> None:
        """Handles agent specific behaviour when episode terminates."""
        pass  # pragma: no cover

    @abstractmethod
    def finalize_episode(self) -> None:
        """Handles agent behaviour when completing an episode, such as decaying an Epsilon policy or updating gradients."""
        pass  # pragma: no cover


class FeatureExtractor(nn.Module, ABC):
    """A base class for all feature extractors."""

    @abstractmethod
    def __init__(self, inputs: ExtractorInputs) -> None:
        super().__init__()

        self.inputs = inputs
        self.out_features: int = None

    @abstractmethod
    def forward(x: torch.Tensor) -> Any:
        """Perform a forward pass through the network."""
        pass  # pragma: no cover


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PyTorchModel:
    """Storage for PyTorch model details."""

    optimizer: Type[optim.Optimizer]
    loss: Type[_Loss]
    policy: Type[FeatureExtractor]
    inputs: Type[ExtractorInputs]
