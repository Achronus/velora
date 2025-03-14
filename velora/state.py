from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

import gymnasium as gym
from sqlmodel import Session

from velora.models.base import RLAgent

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
class AnalyticsState:
    """
    A storage container for the details of a [Comet](https://www.comet.com/) or
    [Weights and Biases](https://wandb.ai/) analytics experiment.

    Parameters:
        project_name (str): the name of the project to add this experiment to
        experiment_name (str, optional): the name of the experiment
        tags (List[str], optional): a list of tags associated with the experiment
    """

    project_name: str
    experiment_name: str | None = None
    tags: List[str] | None = None


@dataclass
class TrainState:
    """
    A storage container for the current state of model training.

    Parameters:
        agent (RLAgent): the agent being trained
        env (gymnasium.Env): the environment used for training
        session (sqlmodel.Session): the current metric database session
        experiment_id (int): the current experiment's unique ID
        total_episodes (int): total number of training episodes
        status (Literal["start", "episode", "step", "complete"], optional): the current stage of training.

            - `start` - before training starts.
            - `episode` - inside the episode loop.
            - `step` - inside the training loop.
            - `complete` - completed training.

        current_ep (int, optional): the current episode index
        current_step (int, optional): the current training timestep
        stop_training (bool, optional): a flag to declare training termination
        record_state (RecordState, optional): the video recording state
        analytics_state (AnalyticsState, optional): the analytics state
    """

    agent: RLAgent
    env: gym.Env
    session: Session
    experiment_id: int
    total_episodes: int
    status: StatusLiteral = "start"
    current_ep: int = 0
    current_step: int = 0
    stop_training: bool = False
    record_state: RecordState | None = None
    analytics_state: AnalyticsState | None = None

    def update(
        self,
        *,
        status: StatusLiteral | None = None,
        current_ep: int | None = None,
        current_step: int | None = None,
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
        """
        self.status = status if status else self.status
        self.current_ep = current_ep if current_ep else self.current_ep
        self.current_step = current_step if current_step else self.current_step

    def analytics_update(self) -> None:
        """
        Updates the analytics state details that are `None` dynamically, using
        the current training state.
        """
        agent_name = self.agent.__class__.__name__
        env_name = self.env.spec.name

        new_state = self.analytics_state

        new_state.experiment_name = (
            new_state.experiment_name
            if new_state.experiment_name
            else f"{agent_name}_{env_name}_{self.total_episodes}ep"
        )

        new_state.tags = new_state.tags if new_state.tags else [agent_name, env_name]

        # Update state
        self.analytics_state = new_state
