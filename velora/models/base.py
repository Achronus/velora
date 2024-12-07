from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

import torch
import torch.nn as nn
from torch.optim import Optimizer

from velora.agent.policy import Policy
from velora.agent.storage import Storage
from velora.agent.value import ValueFunction
from velora.config import Config


class AgentModel(ABC, BaseModel):
    """A base class for Agent models."""

    config: Config
    device: torch.device

    _vf: ValueFunction = PrivateAttr(...)
    _policy: Policy = PrivateAttr(...)
    _config_exclusions: list[str] | None = PrivateAttr(default=None)
    _next_action: int | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def vf(self) -> ValueFunction:
        """Returns the agents value function."""
        return self._vf

    @property
    def policy(self) -> Policy:
        """Returns the agents policy."""
        return self._policy

    @abstractmethod
    def log_progress(
        self, ep_idx: int, log_count: int, *args: Any, **kwargs: Any
    ) -> None:
        """Displays helpful episode logs to the console during training."""
        pass  # pragma: no cover

    @abstractmethod
    def act(self, state: Any) -> int:
        """Gets an agents action based on its policy."""
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


class TorchAgentModel(BaseModel):
    """A base class for PyTorch Agent models.

    Args:
        model (torch.nn.Module | velora.models.AgentModel):
        optimizer (torch.optim.Optimizer):
        loss (torch.nn.Loss):
        storage (velora.agent.Storage):
    """

    model: nn.Module
    optimizer: Optimizer
    loss: nn.Module
    storage: Storage

    model_config = ConfigDict(arbitrary_types_allowed=True)
