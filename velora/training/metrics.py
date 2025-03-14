import json
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal

from velora.metrics.db import get_current_episode

if TYPE_CHECKING:
    from velora.models.config import RLAgentConfig  # pragma: no cover

import numpy as np
import torch
from sqlmodel import Session

from velora.metrics.models import Episode, Experiment, Step


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

    @property
    def latest(self) -> float | int:
        """Gets the latest value."""
        return self.values[-1]

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


class TrainMetrics:
    """
    A utility class for working with and storing training metrics for monitoring
    an agents training performance.
    """

    def __init__(self, session: Session, window_size: int, n_episodes: int) -> None:
        """
        Parameters:
            session (sqlmodel.Session): current metric database session
            window_size (int): moving average window size
            n_episodes (int): total number of training episodes
        """
        self.session = session
        self.window_size = window_size
        self.n_episodes = n_episodes

        self._ep_rewards = MovingMetric(window_size=window_size)
        self._current_losses = StepStorage()

        self.experiment_id: int | None = None

    def start_experiment(self, config: "RLAgentConfig") -> None:
        """
        Confirms the start of a metric experiment by adding it to the database and
        storing its unique ID locally.

        Parameters:
            agent (str): the name of the agent
            env (str): the name of the environment
        """
        exp = Experiment(
            agent=config.agent,
            env=config.env,
            config=config.model_dump_json(),
        )

        self.session.add(exp)
        self.session.commit()

        self.session.refresh(exp)
        self.experiment_id = exp.id

    def add_step(
        self,
        ep_idx: int,
        step_idx: int,
        critic: float,
        actor: float,
        action: torch.Tensor,
        action_threshold: float,
    ) -> None:
        """
        Add timesteps metrics to the metric database.

        Parameters:
            ep_idx (int): the current episode index
            step_idx (int): the current timestep index
            critic (float): critic step loss
            actor (float): actor step loss
            action (torch.Tensor): the agent action for this timestep
            action_threshold (float): explore-exploit action threshold
                (e.g., noise scale)
        """
        self._exp_created_check()

        # is_explore = bool((action.mean().abs() >= action_threshold).item())

        step = Step(
            experiment_id=self.experiment_id,
            episode_id=ep_idx,
            step_num=step_idx,
            action=json.dumps(action.tolist()),
            actor_loss=actor,
            critic_loss=critic,
            # is_exploration=is_explore,
        )
        self.session.add(step)
        self.session.commit()

        self._current_losses.add(
            {
                "critic_losses": critic,
                "actor_losses": actor,
            }
        )

    def add_episode(self, ep_idx: int, reward: float, n_steps: int) -> None:
        """
        Add episode metrics to the metric database and reset step accumulators.

        Parameters:
            ep_idx (int): the current episode index
            reward (float): episode reward
            n_steps (int): number of steps after episode done
        """
        self._exp_created_check()

        self._ep_rewards.add(reward)

        actor_loss = self._current_losses.actor_avg()
        critic_loss = self._current_losses.critic_avg()

        moving_avg = self.reward_moving_avg()
        moving_std = self.reward_moving_std()

        ep = Episode(
            experiment_id=self.experiment_id,
            episode_num=ep_idx,
            reward=reward,
            length=n_steps,
            reward_moving_avg=moving_avg,
            reward_moving_std=moving_std,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
        )
        self.session.add(ep)
        self.session.commit()

        # Reset step storage
        self._current_losses.empty()

    def reward_moving_avg(self) -> float:
        """
        Calculates the average moving reward.

        Returns:
            avg (float): the average moving reward.
        """
        return self._ep_rewards.mean()

    def reward_moving_std(self) -> float:
        """
        Calculates the average reward moving standard deviation.

        Returns:
            avg (float): the average moving standard deviation.
        """
        return self._ep_rewards.std()

    def info(self, current_ep: int) -> None:
        """
        Outputs basic information to the console.

        Parameters:
            current_ep (int): the current episode index
        """
        results = get_current_episode(self.session, self.experiment_id, current_ep)

        for ep in results:
            print(
                f"Episode: {current_ep}/{self.n_episodes}, "
                f"Avg Reward: {ep.reward_moving_avg:.2f}, "
                f"Critic Loss: {ep.critic_loss:.2f}, "
                f"Actor Loss: {ep.actor_loss:.2f}"
            )

    def _exp_created_check(self) -> None:
        """
        Helper method. Performs error handling for checking if an experiment
        has been created first.

        Used in `add_step` and `add_episode`.
        """
        if not self.experiment_id:
            raise RuntimeError(
                "An experiment must be created first!\nCreate one with the '<TrainHandler_instance>.metrics.add_experiment()' method."
            )
