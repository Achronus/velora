from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Self, get_args

import numpy as np
import torch

MetricStateDictKeys = Literal["window_size", "n_episodes", "storage"]
MetricKeys = Literal["ep_rewards", "critic_losses", "actor_losses", "ep_lengths"]

VALID_METRIC_KEYS = set(get_args(MetricKeys))


@dataclass
class StepStorage:
    """
    A storage container for step metrics.

    Useful for calculating the episodic average values to store in `MetricStorage`.

    Attributes:
        critic_losses (List[float]): a list of agent Critic loss values
        actor_losses (List[float]): a list of agent Actor loss values
    """

    critic_losses: List[float] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)

    def critic_avg(self) -> float:
        """
        Computes the critic loss average. Useful for computing episodic averages.

        Returns:
            avg (float): critic loss step average
        """
        if len(self.critic_losses) == 0:
            return 0

        return np.mean(self.critic_losses).item()

    def actor_avg(self) -> float:
        """
        Computes the actor loss average. Useful for computing episodic averages.

        Returns:
            avg (float): actor loss step average
        """
        if len(self.actor_losses) == 0:
            return 0

        return np.mean(self.actor_losses).item()

    def add(self, items: Dict[Literal["critic_losses", "actor_losses"], float]) -> None:
        """
        Stores one or more metrics into storage. Must be single values with
        their respective keys.

        Parameters:
            items (Dict[str, float]): a set of key-value pairs for
                loggable metrics.

                Valid Options -

                - `critic_losses` - agent Critic loss value
                - `actor_losses` - agent Actor loss value
        """
        for key, value in items.items():
            array: List = getattr(self, key)
            array.append(value)

    def empty(self) -> None:
        """Empty storage."""
        self.critic_losses.clear()
        self.actor_losses.clear()


@dataclass
class MovingMetric:
    """
    Tracks a metric with a moving window for statistics.

    Attributes:
        values (List[float | int]): a list of values
        window (collections.deque): a list of values for the statistics
        window_size (int): the window size of the moving statistics.
            Default is `100`
    """

    values: List[float | int] = field(default_factory=list)
    window: deque = field(default_factory=deque)
    window_size: int = 100

    def add(self, value: float | int) -> None:
        """
        Adds a value and updates the window.

        Parameters:
            value (float): value to add
        """
        self.values.append(value)
        self.window.append(value)

        if len(self.window) > self.window_size:
            self.window.popleft()  # Remove oldest value

    def mean(self, values: List[float] | None = None) -> float:
        """
        Calculates the mean of values or the current window.

        Parameters:
            values (List[float], optional): Values to calculate mean for.
                If `None`, uses the current window

        Returns:
            avg (float): the calculated mean.
        """
        values = values if values is not None else self.window
        return np.mean(values).item() if values else 0.0

    def std(self, values: List[float] | None = None) -> float:
        """
        Calculates the standard deviation of values or the current window.

        Parameters:
            values (List[float], optional): Values to calculate standard deviation
                for. If `None`, uses the current window

        Returns:
            std (float): the calculated standard deviation.
        """
        values = values if values is not None else self.window
        return np.std(values).item() if len(values) > 1 else 0.0

    def __len__(self) -> int:
        """Returns the number of items in the values array."""
        return len(self.values)


@dataclass
class SimpleMetricStorage:
    """
    A simple storage container for episodic metrics.

    Attributes:
        ep_rewards (List[float]): episode rewards
        ep_lengths (List[int]): episode lengths
        critic_losses (List[float]): Critic losses
        actor_losses (List[float]): Actor losses
    """

    ep_rewards: List[float] = field(default_factory=list)
    ep_lengths: List[int] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """
        Loads a saved metric state.

        Parameters:
            filepath (str | Path): the location of the saved state

        Returns:
            metrics (SimpleMetricStorage): a new storage container with the saved state.
        """
        load_path = Path(filepath)
        checkpoint: Dict[MetricKeys, List[float | int]] = torch.load(load_path)

        return cls(**checkpoint)


class MetricStorage:
    """
    A storage container for episodic metrics.

    Parameters:
        window_size (int): moving average window size
    """

    def __init__(self, window_size: int) -> None:
        self._ep_rewards = MovingMetric(window_size=window_size)
        self._ep_lengths = MovingMetric(window_size=window_size)
        self._critic_losses = MovingMetric(window_size=window_size)
        self._actor_losses = MovingMetric(window_size=window_size)

        self._current_losses = StepStorage()

    @property
    def ep_rewards(self) -> MovingMetric:
        """
        A storage container for episode rewards with a moving average window.

        Returns:
            rewards (MovingMetric): episode rewards moving metric container.
        """
        return self._ep_rewards

    @property
    def ep_lengths(self) -> MovingMetric:
        """
        A storage container for episode lengths with a moving average window.

        Returns:
            lengths (MovingMetric): episode lengths moving metric container.
        """
        return self._ep_lengths

    @property
    def critic_losses(self) -> MovingMetric:
        """
        A storage container for Critic losses with a moving average window.

        Returns:
            losses (MovingMetric): episode Critic losses moving metric container.
        """
        return self._critic_losses

    @property
    def actor_losses(self) -> MovingMetric:
        """
        A storage container for Actor losses with a moving average window.

        Returns:
            losses (MovingMetric): episode Actor losses moving metric container.
        """
        return self._actor_losses

    def save_state(self) -> Dict[MetricKeys, List[float | int]]:
        """
        Return a dictionary containing the episode storage contents.

        Includes the values for:
        `[ep_rewards, ep_lengths, critic_losses, actor_losses]`.

        Can be loaded back into a `velora.training.SimpleMetricStorage` object.

        Returns:
            state_dict (Dict[str, List[float | int]): a dictionary containing the current state of the storage.
        """
        return {
            k.lstrip("_"): v.values
            for k, v in self.__dict__.items()
            if isinstance(v, MovingMetric)
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ep_rewards={len(self._ep_rewards)}, "
            f"ep_lengths={len(self._ep_lengths)}, "
            f"critic_losses={len(self._ep_lengths)}, "
            f"actor_losses={len(self._ep_lengths)}"
            ")"
        )


class TrainMetrics:
    """
    A utility class for working with training metrics.
    """

    def __init__(self, window_size: int, n_episodes: int) -> None:
        """
        Parameters:
            window_size (int): moving average window size
            n_episodes (int): total number of training episodes
        """
        self.window_size = window_size
        self.n_episodes = n_episodes

        self._storage = MetricStorage(window_size)

    @property
    def n_stored(self) -> int:
        """
        Gets the number of stored values.

        Returns:
            stored (int): the total number of stored episode values.
        """
        return len(self._storage.ep_rewards)

    @property
    def storage(self) -> MetricStorage:
        """
        Gets the metric storage container.

        Returns:
            metrics (MetricStorage): a container with results calculated during training.
        """
        return self._storage

    @property
    def ep_rewards(self) -> List[float]:
        """
        Training episode reward values.

        Returns:
            rewards (List[float]): a list of episode rewards.
        """
        return self.storage.ep_rewards.values

    @property
    def ep_lengths(self) -> List[int]:
        """
        Training episode timestep sizes.

        Returns:
            lengths (List[int]): a list of episode lengths.
        """
        return self.storage.ep_lengths.values

    @property
    def critic_losses(self) -> List[float]:
        """
        Training episode Critic losses.

        Returns:
            losses (List[float]): a list of Critic losses.
        """
        return self.storage.critic_losses.values

    @property
    def actor_losses(self) -> List[float]:
        """
        Training episode Actor losses.

        Returns:
            losses (List[float]): a list of Actor losses.
        """
        return self.storage.actor_losses.values

    def add_step(self, critic: float, actor: float) -> None:
        """
        Stores a critic and actor loss value for the current timestep.

        Parameters:
            critic (float): critic step loss
            actor (float): actor step loss
        """
        self._storage._current_losses.add(
            {"critic_losses": critic, "actor_losses": actor}
        )

    def add_episode(self, reward: float, n_steps: int) -> None:
        """
        Add episode metrics and reset step accumulators.

        Parameters:
            reward (float): episode reward
            n_steps (int): number of steps after episode done
        """
        self._storage._ep_rewards.add(reward)
        self._storage._ep_lengths.add(n_steps)

        # Add episode losses
        actor_loss = self._storage._current_losses.actor_avg()
        critic_loss = self._storage._current_losses.critic_avg()
        self._storage._actor_losses.add(actor_loss)
        self._storage._critic_losses.add(critic_loss)

        # Reset step storage
        self._storage._current_losses.empty()

    def avg_reward(self) -> float:
        """
        Calculates the average episodic reward.

        Returns:
            avg (float): the average reward.
        """
        return self._storage._ep_rewards.mean()

    def info(self, current_ep: int) -> None:
        """
        Outputs basic information to the console.

        Parameters:
            current_ep (int): the current episode index
        """
        avg_reward = self.avg_reward()
        avg_critic_loss = self._storage._critic_losses.mean()
        avg_actor_loss = self._storage._actor_losses.mean()

        print(
            f"Episode: {current_ep}/{self.n_episodes}, "
            f"Avg Reward: {avg_reward:.2f}, "
            f"Critic Loss: {avg_critic_loss:.2f}, "
            f"Actor Loss: {avg_actor_loss:.2f}"
        )

    def save(self, filepath: str | Path) -> None:
        """
        Saves the metric storage values to a file.

        Saved state can only be loaded using
        `velora.training.SimpleMetricStorage.load()`.

        Parameters:
            filepath (str | Path): where to save the metrics
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self._storage.save_state()
        torch.save(state_dict, save_path)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"window_size={self.window_size}, "
            f"n_episodes={self.n_episodes}, "
            f"n_stored={self.n_stored}"
            ")"
        )
