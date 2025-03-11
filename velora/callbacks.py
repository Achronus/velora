import os
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, get_args, override

import gymnasium as gym

if TYPE_CHECKING:
    from velora.state import TrainState  # pragma: no cover

from velora.models.base import RLAgent
from velora.state import AnalyticsState, RecordMethodLiteral, RecordState


class TrainCallback:
    """
    Abstract base class for all training callbacks.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def __call__(self, state: "TrainState") -> "TrainState":
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

    def __call__(self, state: "TrainState") -> "TrainState":
        if state.stop_training:
            return state

        if state.status == "episode":
            reward = state.metrics.reward_moving_avg()

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
        dirname: str,
        *,
        frequency: int = 100,
        buffer: bool = False,
    ) -> None:
        """
        Parameters:
            dirname (str): the model directory name to save checkpoints.
                Automatically created inside `checkpoints` directory as
                `checkpoints/<dirname>/saves`.

                Compliments `TrainCallback.RecordVideos` callback.

            frequency (int, optional): save frequency (in episodes)
            buffer (bool, optional): whether to save the final buffer state
        """
        self.filepath = Path("checkpoints", dirname, "saves")
        self.frequency = frequency
        self.buffer = buffer

        if self.filepath.exists():
            raise FileExistsError(
                f"Items already exist in the '{self.filepath.parent}' directory! Either change the 'dirname' or delete the folders contents."
            )

        print(
            f"'{self.__class__.__name__}' enabled with ep_{frequency=} and {buffer=}."
        )

    def __call__(self, state: "TrainState") -> "TrainState":
        # Only perform checkpoint operations on episode and complete events
        if state.status != "episode" and state.status != "complete":
            return state

        # Create directory if it doesn't exist on first call
        self.filepath.mkdir(parents=True, exist_ok=True)

        ep_idx = state.current_ep

        should_save = False
        filename = f"{state.env.spec.name}_"
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
            self.save_checkpoint(state.agent, ep_idx, filename, buffer)

        return state

    def save_checkpoint(
        self,
        agent: RLAgent,
        ep: int,
        filename: str,
        buffer: bool,
    ) -> None:
        """
        Saves a checkpoint at a given episode with the given suffix.

        Parameters:
            agent (RLAgent): the agent being trained
            ep (int): the current episode index
            filename (str): the checkpoint filename
            buffer (bool): whether to save the buffer state
        """
        checkpoint_path = Path(self.filepath, f"{filename}.pt")

        agent.save(checkpoint_path, buffer=buffer)
        print(f"Checkpoint saved at episode {ep}: {checkpoint_path}")

        if buffer:
            buffer_path = Path(self.filepath, f"{filename}.buffer.pt")
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

        if self.dirpath.exists():
            raise FileExistsError(
                f"Files already exist in the '{self.dirpath.parent}' directory! Either change the 'dirname' or delete/move the folder and its contents."
            )

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

    def __call__(self, state: "TrainState") -> "TrainState":
        # 'start': Set the recording state
        if state.status == "start":
            state.record_state = self.details

            state.env = gym.wrappers.RecordVideo(
                state.env,
                name_prefix=state.env.spec.name,
                **state.record_state.to_wrapper(),
            )

        # Ignore other events
        return state


class CometAnalytics(TrainCallback):
    """
    A callback that enables [`comet-ml`](https://www.comet.com/site/) cloud-based
    analytics tracking.

    Requires Comet ML API key set using the `COMET_API_KEY` environment variable.

    Features:

    - Upload agent configuration objects
    - Track model weights (actor/critic)
    - Tracks episodic training metrics
    - Uploads video recordings (if `RecordVideos` callback applied)
    """

    @override
    def __init__(
        self,
        project_name: str,
        experiment_name: str | None = None,
        *,
        tags: List[str] | None = None,
    ) -> None:
        """
        Parameters:
            project_name (str): the name of the Comet ML project to add this
                experiment to
            experiment_name (str, optional): the name of this experiment run.
                If `None`, automatically creates the name using the format
                `<agent_classname>_<env_name>_<n_episodes>ep`
            tags (List[str], optional): a list of tags associated with the
                experiment. If `None` adds the `agent_classname` and `env_name`
                by default
        """
        try:
            from comet_ml import Experiment
        except ImportError:
            raise ImportError(
                "Failed to load the 'comet_ml' package. Have you installed it using 'pip install velora[comet]'?"
            )

        api_key = os.getenv("COMET_API_KEY", None)
        if api_key is None or api_key == "":
            raise ValueError(
                "Missing 'api_key'! Store it as a 'COMET_API_KEY' environment variable."
            )

        self.experiment: Experiment | None = None
        self.state = AnalyticsState(
            project_name=project_name,
            experiment_name=experiment_name,
            tags=tags,
        )

        experiment_name = experiment_name if experiment_name else "auto"
        print(
            f"'{self.__class__.__name__}' enabled with {project_name=} and {experiment_name=}."
        )

    def __call__(self, state: "TrainState") -> "TrainState":
        # Setup experiment
        if state.status == "start":
            # Update comet training state
            state.analytics_state = self.state
            state.analytics_update()

            self.init_experiment(state)

            # Log config
            self.experiment.log_parameters(state.agent.config.model_dump())

        # Send episodic metrics
        if state.status == "episode":
            reward_low, reward_high = state.metrics.storage.ep_rewards.std_bands()

            self.experiment.log_metrics(
                {
                    "ep_reward": state.metrics.storage.ep_rewards.latest,
                    "ep_length": state.metrics.storage.ep_lengths.latest,
                    "ep_reward_moving_avg": state.metrics.reward_moving_avg(),
                    "ep_reward_moving_upper": reward_high,
                    "ep_reward_moving_lower": reward_low,
                    "actor_loss": state.metrics.storage.actor_losses.latest,
                    "critic_loss": state.metrics.storage.critic_losses.latest,
                },
                epoch=state.current_ep,
            )

        # Finalize training
        if state.status == "complete":
            # Log video recordings
            if state.record_state is not None:
                for video in state.record_state.dirpath.iterdir():
                    self.experiment.log_video(str(video), format="mp4")

            self.experiment.end()

        return state

    def init_experiment(self, state: "TrainState") -> None:
        """Setups up a comet experiment and stores it locally.

        Parameters:
            state (TrainState): the current training state
        """
        from comet_ml import Experiment

        self.experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY", None),
            project_name=state.analytics_state.project_name,
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_output_logging=False,
            log_graph=False,
            display_summary_level=0,
        )

        self.experiment.set_name(state.analytics_state.experiment_name)
        self.experiment.add_tags(state.analytics_state.tags)
