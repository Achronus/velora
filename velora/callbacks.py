from abc import abstractmethod
from pathlib import Path
from typing import get_args, override

from velora.models.base import RLAgent
from velora.state import RecordMethodLiteral, RecordState, TrainState


class TrainCallback:
    """
    Abstract base class for all training callbacks.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def __call__(self, state: TrainState) -> TrainState:
        """
        The callback function that gets called during training.

        Parameters:
            state (TrainState): the current training state

        Returns:
            state (TrainState): the current training state.
        """
        pass  # pragma: no cover


class EarlyStopping(TrainCallback):
    """
    A callback that applies early stopping to the training process.
    """

    @override
    def __init__(self, target: float, *, patience: int = 3) -> None:
        """
        Parameters:
            target (float): average reward target to achieve based on a models
                `window_size`
            patience (int, optional): number of times the threshold needs
                to be met to terminate training
        """
        self.target = target
        self.patience = patience

        self.count = 0

        print(
            f"'{self.__class__.__name__}' enabled with reward_{target=} and {patience=}."
        )

    def __call__(self, state: TrainState) -> TrainState:
        if state.stop_training:
            return state

        if state.status == "episode":
            reward = state.avg_reward

            if reward >= self.target:
                self.count += 1

                if self.count >= self.patience:
                    print(
                        f"Early stopping target reached in {state.current_ep} "
                        "episodes! Training complete."
                    )
                    state.stop_training = True
            else:
                self.count = 0

        return state


class SaveCheckpoints(TrainCallback):
    """
    A callback that applies model state saving checkpoints to the training process.
    """

    @override
    def __init__(
        self,
        agent: RLAgent,
        dirname: str,
        *,
        frequency: int = 100,
        buffer: bool = False,
    ) -> None:
        """
        Parameters:
            agent (RLAgent): the agent to use
            dirname (str): the model directory name to save checkpoints.
                Automatically created inside `checkpoints` directory as
                `checkpoints/<dirname>/saves`.

                Compliments `TrainCallback.RecordVideos` callback.

            frequency (int, optional): save frequency (in episodes)
            buffer (bool, optional): whether to save the final buffer state
        """
        self.agent = agent
        self.filepath = Path("checkpoints", dirname, "saves")
        self.frequency = frequency
        self.buffer = buffer

        print(
            f"'{self.__class__.__name__}' enabled with ep_{frequency=} and {buffer=}."
        )

    def __call__(self, state: TrainState) -> TrainState:
        # Only perform checkpoint operations on episode and complete events
        if state.status != "episode" and state.status != "complete":
            return state

        # Create directory if it doesn't exist on first call
        self.filepath.mkdir(parents=True, exist_ok=True)

        ep_idx = state.current_ep

        should_save = False
        filename = f"{state.env}_"
        buffer = False

        # Save checkpoint at specified frequency
        if state.status == "episode" and ep_idx != state.total_episodes:
            if ep_idx % self.frequency == 0:
                should_save = True
                filename += f"ep{ep_idx}"

        # Perform final checkpoint save
        elif state.status == "complete":
            should_save = True
            filename += "final"
            buffer = self.buffer

        if should_save:
            self.save_checkpoint(ep_idx, filename, buffer)

        return state

    def save_checkpoint(self, ep: int, filename: str, buffer: bool) -> None:
        """
        Saves a checkpoint at a given episode with the given suffix.

        Parameters:
            ep (int): the current episode index
            filename (str): the checkpoint filename
            buffer (bool): whether to save the buffer state
        """
        checkpoint_path = Path(self.filepath, f"{filename}.pt")
        buffer_path = Path(self.filepath, f"{filename}.buffer.pt")

        self.agent.save(checkpoint_path, buffer=buffer)
        print(f"Checkpoint saved at episode {ep}: {checkpoint_path}")

        if buffer:
            print(f"Buffer saved at: {buffer_path}")


class RecordVideos(TrainCallback):
    """
    A callback to enable intermittent environment video recording to visualize
    the agent training progress.

    Requires environment with `render_mode="rgb_array"`.
    """

    @override
    def __init__(
        self,
        method: RecordMethodLiteral,
        dirname: str,
        *,
        frequency: int = 100,
    ) -> None:
        """
        Parameters:
            method (Literal["episode", "step"]): the recording method.
                When `episode` records episodically. When `step` records during
                training steps.
            dirname (str): the model directory name to store the videos.
                Automatically created in `checkpoints` directory as
                `checkpoints/<dirname>/videos`.

                Compliments `TrainCallback.SaveCheckpoints` callback.

            frequency (int, optional): the `episode` or `step` record frequency
        """
        if method not in get_args(RecordMethodLiteral):
            raise ValueError(
                f"'{method=}' is not supported. Choices: '{get_args(RecordMethodLiteral)}'"
            )

        self.method = method
        self.dirpath = Path("checkpoints", dirname, "videos")

        def trigger(t: int) -> bool:
            # Skip first item
            if t == 0:
                return False

            return t % frequency == 0

        self.details = RecordState(
            dirpath=self.dirpath,
            method=method,
            episode_trigger=trigger if method == "episode" else None,
            step_trigger=trigger if method == "step" else None,
        )

        print(f"'{self.__class__.__name__}' enabled with {str(method)}_{frequency=}.")

    def __call__(self, state: TrainState) -> TrainState:
        # 'start': Set the recording state
        if state.status == "start":
            state.record_state = self.details

        # Ignore other events
        return state
