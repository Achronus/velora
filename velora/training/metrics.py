from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Self, get_args

import numpy as np
import torch

MetricStateDictKeys = Literal["window_size", "n_episodes", "storage"]
StorageKeyLiteral = Literal[
    "ep_rewards",
    "critic_losses",
    "actor_losses",
    "td_errors",
    "value_estimates",
    "bellman_residuals",
    "q_values",
    "policy_entropy",
    "steps_per_episode",
]

VALID_STORAGE_KEYS = set(get_args(StorageKeyLiteral))


@dataclass
class MetricStorage:
    """
    A storage container for episodic metrics.

    Attributes:
        ep_rewards (List[float], optional): a list of episode rewards
        critic_losses (List[float], optional): a list of agent Critic loss values
        actor_losses (List[float], optional): a list of agent Actor loss values
        td_errors (List[float]): a list of temporal difference errors
        value_estimates (List[float]): a list of value function estimates
        bellman_residuals (List[float]): a list of bellman equation residuals
        q_values (List[float]): a list of Q-value estimates
        policy_entropy (List[float]): a list of policy entropy values
        steps_per_episode (List[int]): a list for the number of steps taken in each
            episode
    """

    ep_rewards: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)
    value_estimates: List[float] = field(default_factory=list)
    bellman_residuals: List[float] = field(default_factory=list)
    q_values: List[float] = field(default_factory=list)
    policy_entropy: List[float] = field(default_factory=list)
    steps_per_episode: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[StorageKeyLiteral, List[float | int]]:
        """
        Return a dictionary containing the storage contents.

        Returns:
            state_dict (Dict[StorageKeyLiteral, List[float | int]]): a dictionary containing the current state of the storage.
        """
        return self.__dict__


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

        self._storage = MetricStorage()

    @property
    def storage(self) -> MetricStorage:
        """
        Gets the metric storage container.

        Returns:
            metrics (MetricStorage): a container with results calculated during training.
        """
        return self._storage

    def add(self, items: Dict[StorageKeyLiteral, float | int]) -> None:
        """
        Stores one or more metrics into storage. Must be single values with their respective
        keys.

        Parameters:
            items (Dict[str, float | int]): a set of key-value pairs for
                loggable metrics.

                Valid Options -

                - `ep_rewards` - the episode reward after agent takes an action (`float`)
                - `critic_losses` - agent Critic loss value (`float`)
                - `actor_losses` - agent Actor loss value (`float`)
                - `td_errors` - temporal difference error (`float`)
                - `value_estimates` - value function estimate (`float`)
                - `bellman_residuals` - bellman equation residual (`float`)
                - `q_values` - Q-value estimate (`float`)
                - `policy_entropy` - policy entropy value (`float`)
                - `steps_per_episode` - number of steps taken in an episode (`int`)
        """
        invalid_keys = set(items.keys()) - VALID_STORAGE_KEYS
        if invalid_keys:
            raise ValueError(
                f"Invalid log keys: {invalid_keys}. Valid keys: {VALID_STORAGE_KEYS}"
            )

        for key, value in items.items():
            array: List = getattr(self._storage, key)
            array.append(value)

    def avg_reward(self) -> float:
        """
        Computes reward moving average based on the window size.

        Returns:
            reward (float): the average reward.
        """
        if len(self._storage.ep_rewards) >= self.window_size:
            return np.mean(self._storage.ep_rewards[-self.window_size :])

        return 0

    def info(self, current_ep: int) -> None:
        """
        Outputs basic information to the console.

        Parameters:
            current_ep (int): the current episode index
        """
        avg_reward = self.avg_reward()
        avg_critic_loss = np.mean(self._storage.critic_losses)
        avg_actor_loss = np.mean(self._storage.actor_losses)

        print(
            f"Episode: {current_ep}/{self.n_episodes}, "
            f"Avg Reward: {avg_reward:.2f}, "
            f"Critic Loss: {avg_critic_loss:.2f}, "
            f"Actor Loss: {avg_actor_loss:.2f}"
        )

    def new_ws(self, value: int) -> None:
        """
        Sets a new `window_size` based on the given `value`.

        Parameters:
            value (int): a new window size
        """
        self.window_size = value

    def state_dict(self) -> Dict[MetricStateDictKeys, Any]:
        """
        Return a dictionary containing the metrics contents. Includes:

        - `window_size` - the current window size
        - `n_episodes` - the total number of training episodes
        - `storage` - the stored metrics state dict

        Returns:
            state_dict (Dict[MetricStateDictKeys, Any): a dictionary containing the current state of the metrics object.
        """
        return {
            "window_size": self.window_size,
            "n_episodes": self.n_episodes,
            "storage": self.storage.state_dict(),
        }

    def save(self, filepath: str | Path) -> None:
        """
        Saves the metrics state to a file.

        Parameters:
            filepath (str | Path): where to save the metric state
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.state_dict()
        torch.save(state_dict, save_path)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """
        Restores the training metrics from a saved state.

        Parameters:
            filepath (str | Path): metrics state file location

        Returns:
            metrics (Self): a new metric filled instance.
        """
        state_dict: Dict[MetricStateDictKeys, Any] = torch.load(filepath)

        metrics = cls(state_dict["window_size"], state_dict["n_episodes"])
        metrics._storage = MetricStorage(**state_dict["storage"])
        return metrics

    @staticmethod
    def create_filepath(filepath: str | Path) -> Path:
        """
        Updates a given `filepath` and converts it into a `metric` friendly one.

        Parameters:
            filepath (str | Path): a filepath to convert

        Returns:
            path (Path): a metric friendly filepath in the form `<filepath>.metrics.<filepath_ext>`.
        """
        path = Path(filepath)
        extension = path.name.split(".")[-1]
        buffer_name = path.name.replace(extension, f"metrics.{extension}")
        return path.with_name(buffer_name)
