from abc import ABC, abstractmethod
from typing import Iterator, Self
import torch

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from velora.utils.validation import device_validation


class Storage(ABC, BaseModel):
    """A base class for agent storage containers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def add(self) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def __getitem__(self, index: int) -> Self:
        pass  # pragma: no cover

    @abstractmethod
    def __len__(self) -> int:
        pass  # pragma: no cover

    @abstractmethod
    def __repr__(self) -> str:
        pass  # pragma: no cover


class EnvStep(BaseModel):
    """
    A single agent step generated by the environment.

    Args:
        action (int): the action index
        obs (torch.Tensor): the current state of the environment
        reward (float): the reward obtained
    """

    action: int
    obs: torch.Tensor
    reward: float

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        return f"EnvStep(action={self.action}, obs={self.obs}, reward={self.reward})"

    def __str__(self) -> str:
        return f"({self.action}, {self.obs}, {self.reward})"

    def action_tensor(self) -> torch.LongTensor:
        """Returns the action as a torch tensor."""
        return torch.tensor(self.action, dtype=torch.long)

    def reward_tensor(self) -> torch.FloatTensor:
        """Returns the reward as a torch tensor."""
        return torch.tensor(self.reward, dtype=torch.float)


class Rollouts(Storage):
    """A finite sequence of agent trajectories (steps) through the environment."""

    device: str | torch.device = Field("cpu", validate_default=True)

    _steps: list[EnvStep] = PrivateAttr([])

    @field_validator("device")
    def validate_device(cls, device: str | torch.device) -> torch.device:
        return device_validation(device)

    def add(self, step: EnvStep) -> None:
        """Adds an environment step."""
        self._steps.append(step)

    def extend(self, steps: list[EnvStep]) -> None:
        """Adds multiple steps after the latest one."""
        self._steps.extend(steps)

    def actions(self) -> torch.LongTensor:
        """Returns the actions for each trajectory."""
        return torch.stack([t.action_tensor() for t in self._steps]).to(self.device)

    def observations(self) -> torch.Tensor:
        """Returns the observations for each trajectory."""
        return torch.stack([t.obs for t in self._steps]).to(self.device)

    def rewards(self) -> torch.FloatTensor:
        """Returns the rewards for each trajectory."""
        return torch.stack([t.reward_tensor() for t in self._steps]).to(self.device)

    def returns(self, gamma: float = 0.9) -> torch.FloatTensor:
        """
        Computes the discounted return for each trajectory.

        The return G_t at each timestep is calculated as:
        G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γᵏR_{t+k+1}

        Where:
        - G_t is the return at time t
        - R_t is the reward at time t
        - γ (gamma) is the discount factor
        - k is the number of steps into the future

        When iterating, this simplifies to use the future return:
        G_t = R_{t} + γG_{t+1}

        Args:
            gamma: Discount factor (default: 0.9)

        Returns:
            G (torch.FloatTensor): A tensor of discounted returns for each trajectory
        """
        if len(self._steps) == 0:
            return torch.tensor([])

        if gamma == 0:
            return self.rewards()

        rewards = self.rewards()
        n_rewards = len(rewards)
        m_shape = (n_rewards, n_rewards)

        mask = torch.triu(torch.ones(m_shape, device=self.device))
        r_matrix = rewards.expand(m_shape)
        idx_row = torch.arange(n_rewards, device=self.device)

        # Create diagonal discount matrix
        d_matrix = idx_row.expand(m_shape)
        d_matrix = d_matrix - idx_row.unsqueeze(1)
        d_matrix = gamma ** torch.clamp(d_matrix, min=0)

        # Sum each row
        return (mask * r_matrix * d_matrix).sum(dim=1).to(self.device)

    def score(self) -> int:
        """Calculates the total score of the rollouts."""
        return self.rewards().sum().item()

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[EnvStep]:
        return iter(self._steps)

    def __getitem__(self, index: int) -> EnvStep:
        return self._steps[index]

    def __repr__(self) -> str:
        return f"Rollouts(steps={self._steps})"

    def __str__(self) -> str:
        return f"[{", ".join([str(step) for step in self._steps])}]"


class Episodes(Storage):
    """A batch of episodes performed by the agent."""

    _eps: list[Rollouts] = PrivateAttr([])

    def add(self, ep: Rollouts) -> None:
        """Adds an episode to the batch."""
        self._eps.append(ep)

    def scores(self) -> torch.LongTensor:
        """Returns the score for each episode."""
        return torch.tensor([ep.score() for ep in self._eps], dtype=torch.long)

    def observations(self) -> torch.Tensor:
        """Returns a simplifed tensor for all observations in the batch in the shape `(n_steps, obs_size)`."""
        return torch.cat([ep.observations() for ep in self._eps])

    def actions(self) -> torch.Tensor:
        """Returns a flattened tensor for all actions in the batch."""
        return torch.cat([ep.actions() for ep in self._eps])

    def to_list(self) -> list[Rollouts]:
        """Returns the Episodes as a list of Rollouts."""
        return self._eps

    def __len__(self) -> int:
        return len(self._eps)

    def __iter__(self) -> Iterator[Rollouts]:
        return iter(self._eps)

    def __getitem__(self, index: int | slice) -> Self | Rollouts:
        if isinstance(index, slice):
            new_eps = Episodes()
            new_eps._eps = self._eps[index]
            return new_eps

        return self._eps[index]

    def __add__(self, other: Self) -> Self:
        """Combines two Episodes objects using the + operator."""
        if not isinstance(other, Episodes):
            raise NotImplementedError(f"Cannot add '{type(other)}' with 'Episodes()'!")

        combined = Episodes()

        combined._eps = self._eps + other._eps
        return combined

    def __repr__(self) -> str:
        return f"Episodes(eps={self._eps})"

    def __str__(self) -> str:
        return f"[{",\n ".join([str(step) for step in self._eps])}]"


class ReplayBuffer(Storage):
    """"""

    pass
