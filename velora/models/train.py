from typing import List

from velora.callbacks import TrainCallback, TrainState


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
        """Helper method. Runs the callbacks."""
        for cb in self.callbacks:
            self.state = cb(self.state)

    def step(self) -> None:
        """Performs `step` callback event."""
        self.state.update(status="step")
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
