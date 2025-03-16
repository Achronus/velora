import time
from types import TracebackType
from typing import TYPE_CHECKING, List, Self, Type

import gymnasium as gym
from sqlmodel import Session

if TYPE_CHECKING:
    from velora.callbacks import TrainCallback  # pragma: no cover

from velora.gym.wrap import add_core_env_wrappers
from velora.metrics.db import get_db_engine
from velora.models.base import RLAgent
from velora.state import TrainState
from velora.time import ElapsedTime
from velora.training.metrics import TrainMetrics
from velora.utils.capture import record_last_episode


class TrainHandler:
    """
    A context manager for handling an agents training state.
    """

    def __init__(
        self,
        agent: RLAgent,
        env: gym.Env,
        n_episodes: int,
        window_size: int,
        callbacks: List["TrainCallback"] | None,
    ) -> None:
        """
        Parameters:
            agent (RLAgent): the agent being trained
            env (Gymnasium.Env): the environment to train the agent on
            n_episodes (int): the total number of training episodes
            window_size (int): episode window size rate
            callbacks (List[TrainCallback] | None): a list of training callbacks.
                If `None` sets to an empty list
        """
        self.agent = agent
        self.env = env
        self.window_size = window_size
        self.n_episodes = n_episodes
        self.callbacks = callbacks or []

        self.state: TrainState | None = None

        self.start_time = 0.0
        self.train_time: ElapsedTime | None = None

        self.engine = get_db_engine()
        self.session: Session | None = None
        self._metrics: TrainMetrics | None = None

    @property
    def metrics(self) -> TrainMetrics:
        """
        Training metric class instance.

        Returns:
            metrics (TrainMetrics): current training metric state.
        """
        return self._metrics

    def __enter__(self) -> Self:
        """
        Setup the training context, initializing the environment.

        Returns:
            self (Self): the initialized context.
        """
        self.session = Session(self.engine)
        self._metrics = TrainMetrics(
            self.session,
            self.window_size,
            self.n_episodes,
        )
        self._metrics.start_experiment(self.agent.config)

        self.state = TrainState(
            agent=self.agent,
            env=self.env,
            session=self.session,
            total_episodes=self.n_episodes,
            experiment_id=self._metrics.experiment_id,
        )
        self.start_time = time.time()

        self.start()
        self.env = add_core_env_wrappers(self.env, self.agent.device)

        print(
            f"Training started on {self.env.spec.id} for {self.state.total_episodes} episodes."
            f"\nNote: moving averages computed based on window_size={self.window_size}."
            "\n------------"
        )
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
        # self.save_plots()

        self.complete()

        self.env.close()
        self.session.close()

        self.train_time = ElapsedTime.elapsed(self.start_time)
        print(
            "------------",
            f"\nTraining completed after: {self.train_time}.",
        )

    def _run_callbacks(self) -> None:
        """Helper method. Runs the callbacks and updates the training state."""
        for cb in self.callbacks:
            self.state = cb(self.state)

    def start(self) -> None:
        """
        Performs `start` callback event.
        """
        self._run_callbacks()

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

    def episode(self, current_ep: int) -> None:
        """
        Performs `episode` callback event.

        Parameters:
            current_ep (int): the current training episode index
        """
        self.state.update(status="episode", current_ep=current_ep)
        self._run_callbacks()

    def complete(self) -> None:
        """Performs `complete` callback event."""
        self.state.status = "complete"
        self._run_callbacks()

    def stop(self) -> bool:
        """
        Checks if training should be stopped, such as early stopping.

        Returns:
            stop (bool): `True` if training should be stopped, `False` otherwise.
        """
        return self.state.stop_training

    def record_last_episode(self) -> None:
        """
        If recording videos enabled, captures a recording of the last episode.
        """
        if self.state.record_state is not None:
            dirname = self.state.record_state.dirpath.parent.name
            print()
            record_last_episode(self.agent, self.env.spec.id, dirname)
