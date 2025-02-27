from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np

LogKeyLiteral = Literal["ep_rewards", "critic_losses", "actor_losses"]


@dataclass
class TrainMetrics:
    """
    A storage container for agent metrics during training.

    Parameters:
        ep_rewards (List[float], optional): a list of episode rewards
        critic_losses (List[float], optional): a list of agent Critic loss values
        actor_losses (List[float], optional): a list of agent Actor loss values
    """

    ep_rewards: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)


class MetricsTracker:
    """Tracks training metrics."""

    def __init__(self, total_episodes: int, window_size: int) -> None:
        """
        Parameters:
            total_episodes (int): total number of training episodes
            window_size (int): episode rate for calculating reward moving average
        """
        self.total_episodes = total_episodes
        self.window_size = window_size

        self.storage = TrainMetrics()

    def log(self, items: Dict[LogKeyLiteral, float]) -> None:
        """
        Logs one or more metrics to the tracker.

        Parameters:
            items (Dict[LogKeyLiteral, float]): a set of key-value pairs for
                loggable metrics. Valid Options -

                - `ep_rewards` - the episode reward after agent takes an action
                - `critic_losses` - agent Critic loss value
                - `actor_losses` - agent Actor loss value
        """
        for key, value in items.items():
            array: List = getattr(self.storage, key)
            array.append(value)

    def avg_reward(self) -> float:
        """Computes reward moving average based on the window size."""
        return np.mean(self.storage.ep_rewards[-self.window_size :])

    def print(self, current_ep: int) -> None:
        """
        Outputs basic information to the console.

        Parameters:
            current_ep (int): the current episode index
        """
        avg_reward = self.avg_reward()
        avg_critic_loss = np.mean(self.storage.critic_losses)
        avg_actor_loss = np.mean(self.storage.actor_losses)

        print(
            f"Episode: {current_ep}/{self.total_episodes}, "
            f"Avg Reward: {avg_reward:.2f}, "
            f"Critic Loss: {avg_critic_loss:.2f}, "
            f"Actor Loss: {avg_actor_loss:.2f}"
        )
