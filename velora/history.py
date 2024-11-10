from typing import Iterator, Self
import torch

from pydantic import BaseModel, ConfigDict, PrivateAttr


class Trajectory(BaseModel):
    """
    A single agent step generated by the environment.

    Args:
        action (int): the action index
        observation (torch.Tensor): the current state of the environment
        reward (float): the reward obtained
    """

    action: int
    observation: torch.Tensor
    reward: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


class History(BaseModel):
    """A finite sequence of agent trajectories (steps) through the environment."""

    _items: list[Trajectory] = PrivateAttr([])

    def add(self, step: Trajectory) -> None:
        """Adds an environment step to the history."""
        self._items.append(step)

    def extend(self, steps: list[Trajectory]) -> None:
        """Adds multiple steps after the latest one."""
        self._items.extend(steps)

    def empty(self) -> None:
        """Empties the trajectories from the history."""
        self._items.clear()

    def actions(self) -> torch.LongTensor:
        """Returns the actions for each trajectory."""
        return torch.tensor([t.action for t in self._items], dtype=torch.long)

    def observations(self) -> torch.Tensor:
        """Returns the observations for each trajectory."""
        return torch.stack([t.observation for t in self._items])

    def rewards(self) -> torch.LongTensor:
        """Returns the rewards for each trajectory."""
        return torch.tensor([t.reward for t in self._items], dtype=torch.long)

    def returns(self, gamma: float = 0.9) -> torch.FloatTensor:
        """
        Computes the discounted return for each trajectory. Starts at last trajectory and iterates backwards through time `t`.

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
            G: A list of discounted returns for each trajectory
        """
        if len(self._items) == 0:
            return []

        n = len(self._items)
        G = [0.0] * n

        G[n - 1] = self._items[n - 1].reward

        for t in range(n - 2, -1, -1):
            G[t] = self._items[t].reward + gamma * G[t + 1]

        return torch.tensor(G)

    def score(self) -> int:
        """Calculates the total score of the history."""
        return self.rewards().sum().item()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._items)

    def __getitem__(self, index: int) -> Trajectory:
        return self._items[index]


class Episodes(BaseModel):
    """A batch of episodes performed by the agent."""

    _eps: list[History] = PrivateAttr([])

    def add(self, ep: History) -> None:
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

    def __len__(self) -> int:
        return len(self._eps)

    def __iter__(self) -> Iterator[History]:
        return iter(self._eps)

    def __getitem__(self, index: int | slice) -> Self:
        new_eps = Episodes()
        if isinstance(index, slice):
            new_eps._eps = self._eps[index]
        else:
            new_eps._eps = [self._eps[index]]
        return new_eps

    def __add__(self, other: Self) -> Self:
        """Combines two Episodes objects using the + operator."""
        if not isinstance(other, Episodes):
            return NotImplemented

        combined = Episodes()

        combined._eps = self._eps + other._eps
        return combined
