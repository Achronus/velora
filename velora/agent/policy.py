from abc import ABC

import torch
from torch.distributions import Categorical

from pydantic import BaseModel, Field, ConfigDict


class Policy(ABC, BaseModel):
    """A base class for agent policies."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EpsilonPolicy(Policy):
    """
    An Epsilon (ε) policy with ε-Soft and ε-Greedy approaches.

    Args:
        epsilon (float, optional): Exploration probability threshold `[0, 1]` (default is 1)
        min_epsilon (float, optional): Minimum epsilon value possible when decaying. Bound between `[0, 1]` (default is 0.1)
        decay_rate (float, optional): A fixed decay rate for epsilon. Used when calling one of the `decay` methods. Bound between `[0, 1]` (default is 0.01)
        device (torch.device, optional): Device to run computations on, such as `cpu`, `cuda` (default is cpu)
    """

    epsilon: float = Field(default=1, ge=0, le=1)
    min_epsilon: float = Field(default=0.1, ge=0, le=1)
    decay_rate: float = Field(default=0.01, ge=0, le=1)
    device: torch.device = torch.device("cpu")

    def decay_linear(self) -> None:
        """Linearly decays epsilon by the decay rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)

    def decay_exp(self) -> None:
        """Exponentially decays epsilon by the decay rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self.decay_rate))

    def greedy_action(self, q_state: torch.Tensor) -> int:
        """
        Returns an action using an ε-Greedy approach.

        Performs purely random exploration when below ε. Otherwise, selects the action with the highest value (acts greedily).

        Args:
            q_state (torch.Tensor): Action values/Q-values used to determine the best action

        Returns:
            action (int): an action following the ε-Greedy approach
        """
        action_probs = q_state.to(self.device)

        # Exploration: random action
        if torch.rand(1).item() < self.epsilon:
            return int(torch.randint(len(action_probs), (1,)).item())

        # Exploitation: best action
        return int(torch.argmax(action_probs).item())

    def as_dist(self, q_state: torch.Tensor) -> Categorical:
        """
        Converts Q-values to a probability distribution.

        Args:
            q_state (torch.Tensor): Action-values/Q-values to convert to probabilities

        Returns:
            probs (torch.distributions.Categorical): A probability distribution over actions
        """
        action_probs = q_state.to(self.device)
        probs = torch.softmax(action_probs, dim=0)
        return Categorical(probs)

    def soft_dist(self, q_state: torch.Tensor) -> Categorical:
        """
        Returns a Categorical distribution over actions using the ε-Soft exploration strategy.

        Creates a probability distribution where:
        - Each action receives a base probability of ε/n
        - The best action receives an additional (1 - ε) probability

        Args:
            q_state (torch.Tensor): Action values/Q-values used to determine the best action

        Returns:
            probs (torch.distributions.Categorical): A probability distribution over actions
        """
        action_probs = q_state.to(self.device)
        # Base probability for all actions (epsilon/n)
        probs = torch.full_like(action_probs, self.epsilon / len(action_probs))

        # Find best action and add remaining probability to it
        best_action = torch.argmax(action_probs)
        probs[best_action] += 1 - self.epsilon

        return Categorical(probs)

    def soft_action(self, q_state: torch.Tensor) -> int:
        """
        Returns an action using an ε-Soft approach.

        Performs a weighted exploration, giving a higher probability towards the best action.

        Args:
            q_state (torch.Tensor): Action values/Q-values used to determine the best action

        Returns:
            action (int): an action following the ε-Soft approach
        """
        dist = self.soft_dist(q_state)
        return int(dist.sample().item())
