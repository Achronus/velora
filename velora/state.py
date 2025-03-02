from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal

StatusLiteral = Literal["start", "episode", "step", "complete"]
RecordMethodLiteral = Literal["episode", "step"]


@dataclass
class RecordState:
    """
    A storage container for the video recording state.

    Parameters:
        dirpath (Path): the video directory path to store the videos
        method (Literal["episode", "step"]): the recording method
        episode_trigger (Callable[[int], bool], optional): the `episode` recording
            trigger
        step_trigger (Callable[[int], bool], optional): the `step` recording trigger
    """

    dirpath: Path
    method: RecordMethodLiteral
    episode_trigger: Callable[[int], bool] | None = None
    step_trigger: Callable[[int], bool] | None = None

    def to_wrapper(self) -> Dict[str, Any]:
        """
        Converts the state into wrapper parameters.

        Returns:
            params (Dict[str, Any]): values as parameters for [Gymnasium's RecordVideo](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo) wrapper.

            Includes the following keys - `[video_folder, episode_trigger, step_trigger]`.
        """
        return {
            "video_folder": self.dirpath,
            "episode_trigger": self.episode_trigger,
            "step_trigger": self.step_trigger,
        }


@dataclass
class TrainState:
    """
    A storage container for the current state of model training.

    Parameters:
        env (str): the name of the environment to train on
        total_episodes (int): total number of training episodes
        status (Literal["start", "episode", "step", "complete"], optional): the current stage of training.

            - `start` - before training starts.
            - `episode` - inside the episode loop.
            - `step` - inside the training loop.
            - `complete` - completed training.

        current_ep (int, optional): the current episode index
        current_step (int, optional): the current training timestep
        avg_reward (float, optional): the episodes average reward value
        stop_training (bool, optional): a flag to declare training termination
        record_state (RecordState, optional): the video recording state
    """

    env: str
    total_episodes: int
    status: StatusLiteral = "start"
    current_ep: int = 0
    current_step: int = 0
    avg_reward: float = 0
    stop_training: bool = False
    record_state: RecordState | None = None

    def update(
        self,
        *,
        status: StatusLiteral | None = None,
        current_ep: int | None = None,
        current_step: int | None = None,
        avg_reward: float | None = None,
    ) -> None:
        """
        Updates the training state. When any input is `None`, uses existing value.

        Parameters:
            status (Literal["start", "episode", "step", "complete"], optional): the current stage of training.

                - `start` - before training start.
                - `episode` - inside the episode loop.
                - `step` - inside the training loop.
                - `complete` - completed training.

        current_ep (int, optional): the current episode index
        current_step (int, optional): the current training timestep
        avg_reward (float, optional): the episodes average reward value
        """
        self.status = status if status else self.status
        self.current_ep = current_ep if current_ep else self.current_ep
        self.current_step = current_step if current_step else self.current_step
        self.avg_reward = avg_reward if avg_reward else self.avg_reward
