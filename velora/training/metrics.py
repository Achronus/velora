from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from velora.models.config import RLAgentConfig  # pragma: no cover

import torch
from sqlmodel import Session

from velora.metrics.models import Episode, Experiment
from velora.utils.format import number_to_short


class StepStorage:
    """
    A storage container for step metrics.

    Useful for calculating the episodic average values to store in `MetricStorage`.

    Attributes:
        critic_losses (torch.Tensor): a tensor of agent Critic loss values
        actor_losses (torch.Tensor): a tensor of agent Actor loss values
    """

    def __init__(self, capacity: int, *, device: torch.device | None = None) -> None:
        """
        Parameters:
            capacity (int): storage capacity for each tensor
            device (torch.device, optional): the device to perform computations on
        """
        self.capacity = capacity
        self.device = device

        # Position indicators
        self.position = 0
        self.size = 0

        self.critic_losses = torch.zeros((capacity), device=device)
        self.actor_losses = torch.zeros((capacity), device=device)

    def critic_avg(self, ep_length: int) -> torch.Tensor:
        """
        Computes the critic loss average. Useful for computing episodic averages.

        Parameters:
            ep_length (int): size of the episode

        Returns:
            avg (torch.Tensor): critic loss step average
        """
        return self.critic_losses[:ep_length].mean()

    def actor_avg(self, ep_length: int) -> torch.Tensor:
        """
        Computes the actor loss average. Useful for computing episodic averages.

        Parameters:
            ep_length (int): size of the episode

        Returns:
            avg (torch.Tensor): actor loss step average
        """
        return self.actor_losses[:ep_length].mean()

    def add(self, critic: torch.Tensor, actor: torch.Tensor) -> None:
        """
        Adds one of each metric into storage.

        Parameters:
            critic (torch.Tensor): critic loss
            actor (torch.Tensor): actor loss
        """
        self.critic_losses[self.position] = critic.to(self.device)
        self.actor_losses[self.position] = actor.to(self.device)

        # Update position
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def empty(self) -> None:
        """Empty storage."""
        self.critic_losses.zero_()
        self.actor_losses.zero_()

        self.position = 0
        self.size = 0


class MovingMetric:
    """
    Tracks a metric with a moving window for statistics.

    Attributes:
        window (torch.Tensor): a list of values for the statistics
        window_size (int): the window size of the moving statistics
    """

    def __init__(self, window_size: int, *, device: torch.device | None = None) -> None:
        """
        Parameters:
            window_size (int): the size of the moving window
            device (torch.device, optional): the device to perform computations on
        """
        self.window_size = window_size
        self.device = device

        # Position indicators
        self.position = 0
        self.size = 0

        # Pre-allocated storage
        self.window = torch.zeros((window_size), device=device)

    @property
    def latest(self) -> torch.Tensor:
        """Gets the latest value."""
        latest_pos = (self.position - 1) % self.window_size
        return self.window[latest_pos]

    def add(self, value: torch.Tensor) -> None:
        """
        Adds a value and updates the window.

        Parameters:
            value (torch.Tensor): value to add
        """
        self.window[self.position] = value.to(self.device)

        # Update position - deque style
        self.position = (self.position + 1) % self.window_size
        self.size = min(self.size + 1, self.window_size)

    def mean(self) -> torch.Tensor:
        """
        Calculates the mean of values or the current window.

        Returns:
            avg (torch.Tensor): the calculated mean.
        """
        return self.window.mean()

    def std(self) -> torch.Tensor:
        """
        Calculates the standard deviation of values or the current window.

        Returns:
            std (torch.Tensor): the calculated standard deviation.
        """
        return (
            self.window.std()
            if self.window.size(dim=0) > 1
            else torch.tensor(0.0, device=self.device)
        )

    def max(self) -> torch.Tensor:
        """
        Calculates the maximum value of a set of values or the current window.

        Returns:
            max (torch.Tensor): the maximum value.
        """
        return self.window.max()

    def __len__(self) -> int:
        """Returns the number of items in the values array."""
        return self.size


class TrainMetricsBase:
    """
    A base class for training metrics.
    """

    def __init__(
        self,
        session: Session,
        window_size: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            session (sqlmodel.Session): current metric database session
            window_size (int): moving average window size
            device (torch.device, optional): the device to perform computations on
        """
        self.session = session
        self.window_size = window_size
        self.device = device

        self._ep_rewards = MovingMetric(window_size, device=device)
        self._ep_lengths = MovingMetric(window_size, device=device)

        self.experiment_id: int | None = None

        self._critic_loss: torch.Tensor = torch.zeros(1, device=self.device)
        self._actor_loss: torch.Tensor = torch.zeros(1, device=self.device)

    def start_experiment(self, config: "RLAgentConfig") -> None:
        """
        Confirms the start of a metric experiment by adding it to the database and
        storing its unique ID locally.

        Parameters:
            config (RLAgentConfig): an RLAgent config model
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

    def reward_moving_avg(self) -> float:
        """
        Calculates the average moving reward.

        Returns:
            avg (float): the average moving reward.
        """
        return self._ep_rewards.mean().item()

    def reward_moving_std(self) -> float:
        """
        Calculates the average reward moving standard deviation.

        Returns:
            avg (float): the average moving standard deviation.
        """
        return self._ep_rewards.std().item()

    def reward_moving_max(self) -> float:
        """
        Calculates the highest reward for the window.

        Returns:
            max (float): the highest reward in the window.
        """
        return self._ep_rewards.max().item()

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


class EpisodeTrainMetrics(TrainMetricsBase):
    """
    A utility class for working with and storing episodic training metrics for
    monitoring an agents training performance.
    """

    def __init__(
        self,
        session: Session,
        window_size: int,
        n_episodes: int,
        max_steps: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            session (sqlmodel.Session): current metric database session
            window_size (int): moving average window size
            n_episodes (int): total number of training episodes
            max_steps (int): maximum number of steps per episode
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(session, window_size, device=device)

        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self._current_losses = StepStorage(max_steps, device=device)

        self._step_total: torch.Tensor = torch.zeros(
            1,
            dtype=torch.int32,
            device=self.device,
        )

    def add_step(self, critic: torch.Tensor, actor: torch.Tensor) -> None:
        """
        Add timesteps metrics to local storage.

        Parameters:
            critic (torch.Tensor): critic step loss
            actor (torch.Tensor): actor step loss
        """
        self._current_losses.add(critic, actor)

    def add_episode(
        self,
        ep_idx: int,
        reward: torch.Tensor,
        ep_length: torch.Tensor,
    ) -> None:
        """
        Add episode metrics to the metric database and reset step accumulators.

        Parameters:
            ep_idx (int): the current episode index
            reward (torch.Tensor): episode reward
            ep_length (torch.Tensor): number of steps after episode done
        """
        self._exp_created_check()

        self._ep_rewards.add(reward.to(self.device))
        self._ep_lengths.add(ep_length.to(self.device))

        self._actor_loss = self._current_losses.actor_avg(ep_length.item())
        self._critic_loss = self._current_losses.critic_avg(ep_length.item())
        self._step_total += ep_length

        moving_avg = self.reward_moving_avg()
        moving_std = self.reward_moving_std()

        ep = Episode(
            experiment_id=self.experiment_id,
            episode_num=ep_idx,
            reward=reward.item(),
            length=ep_length.item(),
            reward_moving_avg=moving_avg,
            reward_moving_std=moving_std,
            actor_loss=self._actor_loss.item(),
            critic_loss=self._critic_loss.item(),
        )
        self.session.add(ep)
        self.session.commit()

        # Reset step storage
        self._current_losses.empty()

    def info(self, current_ep: int) -> None:
        """
        Outputs basic information to the console.

        Parameters:
            current_ep (int): the current episode index
        """
        ep = number_to_short(current_ep)
        max_eps = number_to_short(self.n_episodes)

        ep_length = number_to_short(int(self._ep_lengths.latest))
        step_total = number_to_short(self._step_total.item())

        max_length = number_to_short(int(self._ep_lengths.max().item()))
        max_steps = number_to_short(self.max_steps)

        print(
            f"Episode: {ep}/{max_eps}, "
            f"Steps: {ep_length}/{step_total}, "
            f"Max Length: {max_length}/{max_steps}, "
            f"Reward Avg: {self.reward_moving_avg():.2f}, "
            f"Reward Max: {self.reward_moving_max():.2f}, "
            f"Critic Loss: {self._critic_loss.item():.2f}, "
            f"Actor Loss: {self._actor_loss.item():.2f}"
        )


class RolloutTrainMetrics(TrainMetricsBase):
    """
    A utility class for working with and storing rollout training metrics for
    monitoring an agents training performance.
    """

    def __init__(
        self,
        session: Session,
        window_size: int,
        n_steps: int,
        total_updates: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            session (sqlmodel.Session): current metric database session
            window_size (int): moving average window size
            n_steps (int): maximum number of training steps
            total_updates (int): the total policy updates to perform
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(session, window_size, device=device)

        self.total_updates = total_updates
        self.n_steps = n_steps

    def add_episode(
        self,
        ep_idx: int,
        reward: torch.Tensor,
        ep_length: torch.Tensor,
        actor_loss: torch.Tensor,
        critic_loss: torch.Tensor,
    ) -> None:
        """
        Adds a rollout's update metrics to the database.

        Parameters:
            ep_idx (int): the current episode index
            reward (torch.Tensor): episode reward
            ep_length (torch.Tensor): number of steps after episode done
        """
        self._exp_created_check()

        self._ep_rewards.add(reward)
        self._ep_lengths.add(ep_length)

        self._actor_loss = actor_loss
        self._critic_loss = critic_loss

        moving_avg = self.reward_moving_avg()
        moving_std = self.reward_moving_std()

        ep = Episode(
            experiment_id=self.experiment_id,
            episode_num=ep_idx,
            reward=reward.item(),
            length=ep_length.item(),
            reward_moving_avg=moving_avg,
            reward_moving_std=moving_std,
            actor_loss=self._actor_loss.item(),
            critic_loss=self._critic_loss.item(),
        )
        self.session.add(ep)
        self.session.commit()

    def info(self, update_idx: int) -> None:
        """
        Outputs basic information to the console.

        Parameters:
            update_idx (int): the current update index
        """
        update_idx = number_to_short(update_idx)
        total_updates = number_to_short(self.total_updates)

        print(
            f"Updates: {update_idx}/{total_updates}, "
            f"Reward Avg: {self.reward_moving_avg():.2f}, "
            f"Reward Max: {self.reward_moving_max():.2f}, "
            f"Critic Loss: {self._critic_loss.item():.2f}, "
            f"Actor Loss: {self._actor_loss.item():.2f}"
        )
