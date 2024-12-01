from abc import ABC

import torch
from torch.distributions import Categorical

from pydantic import BaseModel, Field, ConfigDict, field_validator

from velora.utils.validation import device_validation


class Policy(ABC, BaseModel):
    """A base class for agent policies."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EpsilonPolicy(Policy):
    """
    An Epsilon (ε) policy with ε-Soft and ε-Greedy approaches.

    Args:
        epsilon (float, optional): Exploration probability threshold `[0, 1]` (default: 1)
        min_epsilon (float, optional): Minimum epsilon value possible when decaying (default: 0.1)
        decay_rate (float, optional): A fixed decay rate for epsilon. Used when calling one of the `decay` methods (default: 0.01)
        device (str | torch.device, optional): Device to run computations on, such as `cpu`, `cuda` (default: cpu)
    """

    epsilon: float = Field(1, ge=0, le=1)
    min_epsilon: float = Field(0.1, ge=0, le=1)
    decay_rate: float = 0.01
    device: str | torch.device = Field("cpu", validate_default=True)

    @field_validator("device")
    def validate_device(cls, device: str | torch.device) -> torch.device:
        return device_validation(device)

    def decay_linear(self) -> None:
        """Linearly decays epsilon by the decay rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)

    def decay_exp(self) -> None:
        """Exponentially decays epsilon by the decay rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self.decay_rate))

    def greedy_action(self, actions: torch.Tensor) -> int:
        """
        Returns an action using an ε-Greedy approach.

        Performs purely random exploration when below ε. Otherwise, selects the action with the highest value (acts greedily).

        Args:
            actions (torch.Tensor): Action values/Q-values used to determine the best action

        Returns:
            action (int): an action following the ε-Greedy approach
        """
        actions = actions.to(self.device)

        # Exploration: random action
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(len(actions), (1,)).item()

        # Exploitation: best action
        return torch.argmax(actions).item()

    def soft_probs(self, actions: torch.Tensor) -> Categorical:
        """
        Returns a Categorical distribution over actions using the ε-Soft exploration strategy.

        Creates a probability distribution where:
        - Each action receives a base probability of ε/n
        - The best action receives an additional (1 - ε) probability

        Args:
            actions (torch.Tensor): Action values/Q-values used to determine the best action

        Returns:
            probs (Categorical): A probability distribution over actions
        """
        actions = actions.to(self.device)
        # Base probability for all actions (epsilon/n)
        probs = torch.full_like(actions, self.epsilon / len(actions))

        # Find best action and add remaining probability to it
        best_action = torch.argmax(actions)
        probs[best_action] += 1 - self.epsilon

        return Categorical(probs)

    def soft_action(self, actions: torch.Tensor) -> int:
        """
        Returns an action using an ε-Soft approach.

        Performs a weighted exploration, giving a higher probability towards the best action.

        Args:
            actions (torch.Tensor): Action values/Q-values used to determine the best action

        Returns:
            action (int): an action following the ε-Soft approach
        """
        dist = self.soft_probs(actions)
        return dist.sample().item()
