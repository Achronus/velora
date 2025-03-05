import time
from types import TracebackType
from typing import List, Self, Type

import gymnasium as gym

from velora.callbacks import TrainCallback
from velora.gym.wrap import add_core_env_wrappers
from velora.models.base import RLAgent
from velora.state import TrainState
from velora.time import ElapsedTime
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
        callbacks: List[TrainCallback] | None,
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
        self.callbacks = callbacks or []

        self.state = TrainState(
            agent=agent,
            env=env.spec.name,
            total_episodes=n_episodes,
        )

        self.start_time = 0.0
        self.train_time: ElapsedTime | None = None

    def __enter__(self) -> Self:
        """
        Setup the training context, initializing the environment.

        Returns:
            self (Self): the initialized context.
        """
        self.start_time = time.time()

        self.start()
        self.env = add_core_env_wrappers(self.env, self.agent.device)

        print(
            f"Training started on {self.env.spec.id} for {self.state.total_episodes} episodes."
            f"\nNote: moving averages computed based on window_size={self.window_size}."
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
        self.complete()
        self.env.close()

        self.record_last_episode()

        self.train_time = ElapsedTime.elapsed(self.start_time)
        print(f"Training completed after: {self.train_time}.")

    def _run_callbacks(self) -> None:
        """Helper method. Runs the callbacks and updates the training state."""
        for cb in self.callbacks:
            self.state = cb(self.state)

    def start(self) -> None:
        """
        Performs `start` callback event.
        """
        self._run_callbacks()

        # Wrap environment if record video callback
        if self.state.record_state:
            self.env = gym.wrappers.RecordVideo(
                self.env,
                name_prefix=self.state.env,
                **self.state.record_state.to_wrapper(),
            )

    def step(self, current_step: int) -> None:
        """Performs `step` callback event."""
        self.state.update(status="step", current_step=current_step)
        self._run_callbacks()

    def episode(self, current_ep: int, avg_reward: float) -> None:
        """
        Performs `episode` callback event.

        Parameters:
            current_ep (int): the current episode index
            avg_reward (float): the episodes average reward
        """
        self.state.update(
            status="episode",
            current_ep=current_ep,
            avg_reward=avg_reward,
        )
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
        """If recording videos enabled, captures a recording of the last episode."""
        if self.state.record_state is not None:
            dirname = self.state.record_state.dirpath.parent.name
            record_last_episode(self.agent, self.env.spec.id, dirname)
