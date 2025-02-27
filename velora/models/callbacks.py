from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

from velora.models.base import RLAgent

StatusLiteral = Literal["episode", "step", "complete"]


@dataclass
class TrainState:
    """
    A storage container for the current state of model training.

    Parameters:
        total_episodes (int): total number of training episodes
        status (Literal["episode", "step", "complete"], optional): the current stage of training.

            - `episode` - inside the episode loop.
            - `step` - inside the training loop.
            - `complete` - completed training.
        current_ep (int, optional): the current episode index
        avg_reward (float, optional): the episodes average reward value
    """

    total_episodes: int
    status: StatusLiteral = "episode"
    current_ep: int = 0
    avg_reward: float = 0
    stop_training: bool = False

    def update(
        self,
        *,
        status: StatusLiteral | None = None,
        ep: int | None = None,
        avg_reward: float | None = None,
    ) -> None:
        """
        Updates the training state. When any input is `None`, uses existing value.

        Parameters:
            status (Literal["episode", "step", "complete"], optional): the current stage of training.

                - `episode` - inside the episode loop.
                - `step` - inside the training loop.
                - `complete` - completed training.
        current_ep (int, optional): the current episode index
        avg_reward (float, optional): the episodes average reward value
        """
        self.status = status if status else self.status
        self.current_ep = ep if ep else self.current_ep
        self.avg_reward = avg_reward if avg_reward else self.avg_reward


class TrainCallback:
    """
    Abstract base class for all training callbacks.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def __call__(self, *args, **kwargs) -> TrainState:
        pass  # pragma: no cover


class EarlyStopping(TrainCallback):
    """
    A context manager that applies early stopping to the training process.
    """

    @override
    def __init__(self, target: float, patience: int = 3) -> None:
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

    @override
    def __call__(self, state: TrainState) -> TrainState:
        """
        The callback function that gets called during training.

        Parameters:
            state (TrainState): the current training state

        Returns:
            state (TrainState): the current training state or an updated one if patience achieved.
        """
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
    A context manager that applies model state saving checkpoints to the training process.
    """

    @override
    def __init__(
        self,
        agent: RLAgent,
        prefix: str,
        dirname: str,
        *,
        frequency: int = 100,
        buffer: bool = False,
    ) -> None:
        """
        Parameters:
            agent (RLAgent): the agent to use
            prefix (str): a name applied to the start of checkpoint filenames,
                such as the `gym.Environment` name
            dirname (str): the model directory name to save checkpoints
                Automatically created inside `checkpoints` directory.
            frequency (int, optional): save frequency (in episodes)
            buffer (bool, optional): whether to save the final buffer state
        """
        self.agent = agent
        self.name_prefix = prefix
        self.filepath = Path("checkpoints", dirname)
        self.frequency = frequency
        self.buffer = buffer

    @override
    def __call__(self, state: TrainState) -> TrainState:
        """
        The callback function that gets called during training.

        Parameters:
            state (TrainState): the current training state

        Returns:
            state (TrainState): the current training state.
        """
        # Only perform checkpoint operations on episode events
        if state.status != "episode" and state.status != "complete":
            return state

        # Create directory if it doesn't exist on first call
        self.filepath.mkdir(parents=True, exist_ok=True)

        ep_idx = state.current_ep

        should_save = False
        str_suffix = None
        buffer = False

        # Save checkpoint at specified frequency
        if state.status == "episode" and ep_idx != state.total_episodes:
            if ep_idx % self.frequency == 0:
                should_save = True
                str_suffix = f"ep{ep_idx}"

        # Perform final checkpoint save
        elif state.status == "complete":
            should_save = True
            str_suffix = "final"
            buffer = self.buffer

        if should_save:
            self.save_checkpoint(ep_idx, str_suffix, buffer)

        return state

    def save_checkpoint(self, ep: int, suffix: str, buffer: bool) -> None:
        """
        Saves a checkpoint at a given episode with the given suffix.

        Parameters:
            ep (int): the current episode index
            suffix (str): the suffix to add to the filename
            buffer (bool): whether to save the buffer state
        """
        checkpoint_path = Path(self.filepath, f"{self.name_prefix}_{suffix}.pt")
        buffer_path = Path(self.filepath, f"{self.name_prefix}_{suffix}.buffer.pt")

        self.agent.save(checkpoint_path, buffer=buffer)
        print(f"Checkpoint saved at episode {ep}: {checkpoint_path}")

        if buffer:
            print(f"Buffer saved at: {buffer_path}")
