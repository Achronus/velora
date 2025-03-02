from typing import List

import gymnasium as gym

from velora.callbacks import TrainCallback
from velora.models.base import RLAgent
from velora.state import TrainState
from velora.utils.capture import record_last_episode


class StateHandler:
    """
    A utility class for handling an agents training state.

    Useful for running callback methods and updating the training state
    simultaneously.
    """

    def __init__(
        self,
        env_name: str,
        n_episodes: int,
        callbacks: List[TrainCallback],
    ) -> None:
        """
        Parameters:
            env_name (str): the name of the environment
            n_episodes (int): the total number of training episodes
            callbacks (List[TrainCallback]): a list of training callbacks
        """
        self.callbacks = callbacks

        self.state = TrainState(env=env_name, total_episodes=n_episodes)

    def _run_callbacks(self) -> None:
        """Helper method. Runs the callbacks and updates the training state."""
        for cb in self.callbacks:
            self.state = cb(self.state)

    def start(self, env: gym.Env) -> gym.Env:
        """
        Performs `start` callback event.

        Parameters:
            env (gym.Env): the Gymnasium environment used during training

        Returns:
            env (gym.Env): the same or a newly wrapped environment.
        """
        self._run_callbacks()

        if self.state.record_state:
            env = gym.wrappers.RecordVideo(
                env,
                name_prefix=self.state.env,
                **self.state.record_state.to_wrapper(),
            )

        return env

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
        Checks if training should be stopped.

        Returns:
            stop (bool): `True` if training should be stopped, `False` otherwise.
        """
        return self.state.stop_training

    def record_last(self, agent: RLAgent, env_name: str) -> None:
        """
        If recording videos, captures a recording of the last episode.

        Parameters:
            agent (RLAgent): the agent to use for the recording
            env_name (str): the name of the environment to use for recording
        """
        if self.state.record_state is not None:
            dirname = self.state.record_state.dirpath.parent.name
            record_last_episode(agent, env_name, dirname)
