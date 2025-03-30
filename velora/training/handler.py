import time
from types import TracebackType
from typing import TYPE_CHECKING, List, Self, Type

import gymnasium as gym
from sqlmodel import Session

from velora.utils.format import number_to_short

if TYPE_CHECKING:
    from velora.callbacks import TrainCallback  # pragma: no cover

from velora.gym.wrap import add_core_env_wrappers
from velora.metrics.db import get_db_engine
from velora.models.base import RLAgent
from velora.state import TrainState
from velora.time import ElapsedTime
from velora.training.metrics import (
    EpisodeTrainMetrics,
    RolloutTrainMetrics,
    TrainMetricsBase,
)
from velora.utils.capture import record_last_episode


class TrainHandlerBase:
    """
    A base class for train handlers.
    """

    def __init__(
        self,
        agent: RLAgent,
        env: gym.Env | gym.vector.VectorEnv,
        window_size: int,
        callbacks: List["TrainCallback"] | None,
    ) -> None:
        """
        Parameters:
            agent (RLAgent): the agent being trained
            env (gym.Env | gym.vector.VectorEnv): the environment (or vectorized
                envs) to train the agent on
            window_size (int): episode window size rate
            callbacks (List[TrainCallback] | None): a list of training callbacks.
                If `None` sets to an empty list
        """
        self.agent = agent
        self.env = env
        self.window_size = window_size
        self.callbacks = callbacks or []
        self.device = self.agent.device

        self.state: TrainState | None = None

        self.start_time = 0.0
        self.train_time: ElapsedTime | None = None

        self.engine = get_db_engine()
        self.session: Session | None = None
        self._metrics: TrainMetricsBase | None = None

    def __enter__(self) -> Self:
        """
        Setup the training context, initializing the environment.

        Returns:
            self (Self): the initialized context.
        """
        self.start_time = time.time()

        self.start()
        self.env = add_core_env_wrappers(self.env, self.agent.device)

        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Clean up resources and finalize training.

        Parameters:
            exc_type (Type[BaseException], optional): the exception class, if an
                exception was raised inside the `with` block. `None` otherwise
            exc_val (BaseException, optional): the exception instance, if an
                exception is raised. `None` otherwise
            exc_tb (TracebackType, optional): the traceback object, if an exception
                occurred. `None` otherwise
        """
        self.record_last_episode()

        self.complete()

        self.env.close()
        self.session.close()

        self.train_time = ElapsedTime.elapsed(self.start_time)

        early_stop_str = "Early stopping target reached!\n" if self.stop() else ""
        print(
            "---------------------------------\n"
            f"{early_stop_str}"
            "Training completed in: "
            f"{number_to_short(self.state.current_ep)} episodes, "
            f"{number_to_short(self.state.current_step)} steps, "
            f"and {self.train_time}."
        )

    def _run_callbacks(self) -> None:
        """Helper method. Runs the callbacks and updates the training state."""
        for cb in self.callbacks:
            self.state = cb(self.state)

    def complete(self) -> None:
        """Performs `complete` callback event."""
        self.state.status = "complete"
        self._run_callbacks()

    def start(self) -> None:
        """
        Performs `start` callback event.
        """
        self._run_callbacks()

    def stop(self) -> bool:
        """
        Checks if training should be stopped, such as early stopping.

        Returns:
            stop (bool): `True` if training should be stopped, `False` otherwise.
        """
        return self.state.stop_training

    def episode(self, current_ep: int, ep_reward: float) -> None:
        """
        Performs `episode` callback event.

        Parameters:
            current_ep (int): the current training episode index
            ep_reward (float): the episodes reward (return)
        """
        self.state.update(
            status="episode",
            current_ep=current_ep,
            ep_reward=ep_reward,
        )
        self._run_callbacks()

    def record_last_episode(self) -> None:
        """
        If recording videos enabled, captures a recording of the last episode.
        """
        if self.state.record_state is not None:
            dirname = self.state.record_state.dirpath.parent.name
            print()
            record_last_episode(self.agent, self.env.spec.id, dirname)


class TrainHandler(TrainHandlerBase):
    """
    A context manager for handling an agents training state. Compatible with single
    environments.
    """

    def __init__(
        self,
        agent: RLAgent,
        env: gym.Env,
        n_episodes: int,
        max_steps: int,
        window_size: int,
        callbacks: List["TrainCallback"] | None,
    ) -> None:
        """
        Parameters:
            agent (RLAgent): the agent being trained
            env (gym.Env): the environment to train the agent on
            n_episodes (int): the total number of training episodes
            max_steps (int): maximum number of steps in an episode
            window_size (int): episode window size rate
            callbacks (List[TrainCallback] | None): a list of training callbacks.
                If `None` sets to an empty list
        """
        super().__init__(agent, env, window_size, callbacks)

        self.n_episodes = n_episodes
        self.max_steps = max_steps

    @property
    def metrics(self) -> EpisodeTrainMetrics:
        """
        Training metric class instance.

        Returns:
            metrics (EpisodeTrainMetrics): current training metric state.
        """
        return self._metrics

    def __enter__(self) -> Self:
        """
        Setup the training context, initializing the environment.

        Returns:
            self (Self): the initialized context.
        """
        self.session = Session(self.engine)
        self._metrics = EpisodeTrainMetrics(
            self.session,
            self.window_size,
            self.n_episodes,
            self.max_steps,
            device=self.device,
        )
        self._metrics.start_experiment(self.agent.config)

        self.state = TrainState(
            agent=self.agent,
            env=self.env,
            session=self.session,
            total_episodes=self.n_episodes,
            experiment_id=self._metrics.experiment_id,
        )

        return super().__enter__()

    def start(self) -> None:
        super().start()

        # Update environment with callback wrappers
        self.env = self.state.env

    def step(self, current_step: int) -> None:
        """
        Performs `step` callback event.

        Parameters:
            current_step (int): the current training timestep index
        """
        self.state.update(status="step", current_step=current_step)
        self._run_callbacks()


class VecTrainHandler(TrainHandlerBase):
    """
    A context manager for handling an agents training state. Compatible with
    vectorized environments.
    """

    def __init__(
        self,
        agent: RLAgent,
        envs: gym.vector.VectorEnv,
        n_steps: int,
        batch_size: int,
        window_size: int,
        callbacks: List["TrainCallback"] | None,
    ) -> None:
        """
        Parameters:
            agent (RLAgent): the agent being trained
            env (gym.vector.VectorEnv): the vectorized environments to train
                the agent on
            n_steps (int): maximum number of training steps
            batch_size (int): number of samples per mini-batch
            window_size (int): episode window size rate
            callbacks (List[TrainCallback] | None): a list of training callbacks.
                If `None` sets to an empty list
        """
        super().__init__(agent, envs, window_size, callbacks)

        self.n_steps = n_steps
        self.batch_size = batch_size

        self.total_updates = n_steps // batch_size

        # Setup evaluation environment
        self.eval_env = gym.make(envs.spec.id, render_mode="rgb_array")

    @property
    def metrics(self) -> RolloutTrainMetrics:
        """
        Training metric class instance.

        Returns:
            metrics (RolloutTrainMetrics): current training metric state.
        """
        return self._metrics

    def start(self) -> None:
        super().start()

        # Update eval environment with callback wrappers
        self.eval_env = self.state.env
        self.eval_env = add_core_env_wrappers(self.eval_env, self.device)

    def __enter__(self) -> Self:
        """
        Setup the training context, initializing the environment.

        Returns:
            self (Self): the initialized context.
        """
        self.session = Session(self.engine)
        self._metrics = RolloutTrainMetrics(
            self.session,
            self.window_size,
            self.n_steps,
            self.total_updates,
            device=self.device,
        )
        self._metrics.start_experiment(self.agent.config)

        self.state = TrainState(
            agent=self.agent,
            env=self.eval_env,
            session=self.session,
            total_episodes=self.total_updates,
            experiment_id=self._metrics.experiment_id,
        )

        return super().__enter__()

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)

        self.eval_env.close()

    def increment_step(self, current_step: int) -> None:
        """
        Increments the training states step index.

        Parameters:
            current_step (int): the current training timestep index
        """
        self.state.update(current_step=current_step)
