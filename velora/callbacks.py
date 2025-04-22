import os
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, get_args

try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # pragma: no cover

import gymnasium as gym

if TYPE_CHECKING:
    from velora.state import TrainState  # pragma: no cover
    from velora.models.base import RLModuleAgent  # pragma: no cover

from velora.metrics.db import get_current_episode
from velora.state import AnalyticsState, RecordMethodLiteral, RecordState
from velora.utils.format import number_to_short


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

    @abstractmethod
    def config(self) -> Tuple[str, Dict[str, Any]]:
        """
        Retrieves callback details in the form: `(name, values)`.

        Returns:
            name (str): callback name.
            values (Dict[str, Any]): a dictionary containing callback settings.
        """
        pass  # pragma: no cover

    @abstractmethod
    def info(self) -> str:
        """
        Provides details with basic information about the callback initialization.

        Returns:
            details (str): a string of information.
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
            target (float): episode reward target to achieve
            patience (int, optional): number of times the threshold needs
                to be met to terminate training
        """
        self.target = target
        self.patience = patience

        self.count = 0

    def __call__(self, state: "TrainState") -> "TrainState":
        if state.stop_training:
            return state

        if state.status == "episode":
            if state.ep_reward >= self.target:
                self.count += 1

                if self.count >= self.patience:
                    state.stop_training = True
            else:
                self.count = 0

        return state

    def config(self) -> Tuple[str, Dict[str, Any]]:
        return self.__class__.__name__, {
            "target": self.target,
            "patience": self.patience,
        }

    def info(self) -> str:
        return f"'{self.__class__.__name__}' enabled with 'target={self.target}' and 'patience={self.patience}'."


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

            frequency (int, optional): save frequency (in episodes or steps)
            buffer (bool, optional): whether to save the buffer state
        """
        self.dirname = dirname
        self.frequency = frequency
        self.buffer = buffer

        self.filepath = Path("checkpoints", self.dirname, "saves")

        if self.filepath.exists():
            raise FileExistsError(
                f"Items already exist in the '{self.filepath.parent}' directory! Either change the 'dirname' or delete the folders contents."
            )

    def __call__(self, state: "TrainState") -> "TrainState":
        if state.status == "start":
            state.saving_enabled = True
            state.checkpoint_dir = self.filepath

        # Only perform checkpoint operations on episode and complete events
        if state.status != "episode" and state.status != "complete":
            return state

        # Create directory if it doesn't exist on first call
        self.filepath.mkdir(parents=True, exist_ok=True)

        ep_idx = state.current_ep

        should_save = False
        filename = f"{state.env.spec.name}_"

        # Save checkpoint at specified frequency
        if state.status == "episode" and ep_idx != state.total_episodes:
            if ep_idx % self.frequency == 0:
                should_save = True
                filename += f"{number_to_short(ep_idx)}"

        # Perform final checkpoint save
        elif state.status == "complete":
            should_save = True
            filename += "final"

        if should_save:
            self.save_checkpoint(state.agent, filename)

        return state

    def save_checkpoint(self, agent: "RLModuleAgent", dirname: str) -> None:
        """
        Saves a checkpoint at a given episode with the given suffix.

        Parameters:
            agent (RLModuleAgent): the agent being trained
            dirname (str): the checkpoint directory name
        """
        checkpoint_path = Path(self.filepath, dirname)
        agent.save(checkpoint_path, buffer=self.buffer, config=True)

    def config(self) -> Tuple[str, Dict[str, Any]]:
        return self.__class__.__name__, {
            "dirname": self.dirname,
            "frequency": self.frequency,
            "buffer": self.buffer,
        }

    def info(self) -> str:
        return (
            f"'{self.__class__.__name__}' enabled with 'frequency={self.frequency}' and 'buffer={self.buffer}'.\n"
            f"    Files will be saved in the 'checkpoints/{self.dirname}/saves' directory."
        )


class RecordVideos(TrainCallback):
    """
    A callback to enable intermittent environment video recording to visualize
    the agent training progress.

    Requires environment with `render_mode="rgb_array"`.
    """

    @override
    def __init__(
        self,
        dirname: str,
        *,
        method: RecordMethodLiteral = "episode",
        frequency: int = 100,
        force: bool = False,
    ) -> None:
        """
        Parameters:
            dirname (str): the model directory name to store the videos.
                Automatically created in `checkpoints` directory as
                `checkpoints/<dirname>/videos`.

                Compliments `TrainCallback.SaveCheckpoints` callback.

            method (Literal["episode", "step"], optional): the recording method.
                When `episode` records episodically. When `step` records during
                training steps.
            frequency (int, optional): the `episode` or `step` record frequency
            force (bool, optional): enables file overwriting, ignoring existing
                video files. Useful for continuing model training
        """
        if method not in get_args(RecordMethodLiteral):
            raise ValueError(
                f"'{method=}' is not supported. Choices: '{get_args(RecordMethodLiteral)}'"
            )

        self.dirname = dirname
        self.method = method
        self.frequency = frequency
        self.force = force

        self.dirpath = Path("checkpoints", dirname, "videos")

        if not force and self.dirpath.exists():
            raise FileExistsError(
                f"Files already exist in the '{self.dirpath.parent}' directory! Either change the 'dirname', delete/move the folder and its contents, or use 'force=True' to allow overwriting."
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

    def config(self) -> Tuple[str, Dict[str, Any]]:
        return self.__class__.__name__, {
            "dirname": self.dirname,
            "method": self.method,
            "frequency": self.frequency,
            "force": self.force,
        }

    def info(self) -> str:
        return (
            f"'{self.__class__.__name__}' enabled with 'method={str(self.method)}' and 'frequency={self.frequency}'.\n"
            f"    Files will be saved in the 'checkpoints/{self.dirname}/videos' directory."
        )


class CometAnalytics(TrainCallback):
    """
    A callback that enables [`comet-ml`](https://www.comet.com/site/) cloud-based
    analytics tracking.

    Requires Comet ML API key set using the `COMET_API_KEY` environment variable.

    Features:

    - Upload agent configuration objects
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
        experiment_key: str | None = None,
    ) -> None:
        """
        Parameters:
            project_name (str): the name of the Comet ML project to add this
                experiment to
            experiment_name (str, optional): the name of this experiment run.
                If `None`, automatically creates the name using the format
                `<agent_classname>_<env_name>_<n_episodes>ep`. Ignored when
                using `experiment_key`
            tags (List[str], optional): a list of tags associated with the
                experiment. If `None` adds the `agent_classname` and `env_name`
                by default. Ignored when using `experiment_key`
            experiment_key (str, optional): an existing Comet ML experiment key.
                Used for continuing an experiment
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
        self.experiment_key = experiment_key

        self.state = AnalyticsState(
            project_name=project_name,
            experiment_name=experiment_name,
            tags=tags,
        )

        self.project_name = project_name
        self.experiment_name = experiment_name if experiment_name else "auto"
        self.tags = tags if tags else "auto"

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
        if state.status == "logging":
            if state.logging_type == "episode":
                self.log_episode_data(state)

        # Finalize training
        if state.status == "complete":
            # Log video recordings
            if state.record_state is not None:
                for video in state.record_state.dirpath.iterdir():
                    self.experiment.log_video(str(video), format="mp4")

            self.experiment.end()

        return state

    def log_episode_data(self, state: "TrainState") -> None:
        """
        Logs episodic data to the experiment.

        Parameters:
            state (TrainState): the current training state
        """
        results = get_current_episode(
            state.session,
            state.experiment_id,
            state.current_ep,
        )

        for item in results:
            reward_low = item.reward_moving_avg - item.reward_moving_std
            reward_high = item.reward_moving_avg + item.reward_moving_std

            self.experiment.log_metrics(
                {
                    "reward/moving_lower": reward_low,
                    "reward/moving_avg": item.reward_moving_avg,
                    "reward/moving_upper": reward_high,
                    "episode/return": item.reward,
                    "episode/length": item.length,
                    "losses/actor": item.actor_loss,
                    "losses/critic": item.critic_loss,
                    "losses/entropy": item.entropy_loss,
                },
                epoch=state.current_ep,
            )

    def init_experiment(self, state: "TrainState") -> None:
        """Setups up a comet experiment and stores it locally.

        Parameters:
            state (TrainState): the current training state
        """
        from comet_ml import ExistingExperiment, Experiment

        if self.experiment_key is not None:
            # Continue existing experiment
            self.experiment = ExistingExperiment(
                api_key=os.getenv("COMET_API_KEY", None),
                project_name=state.analytics_state.project_name,
                experiment_key=self.experiment_key,
                disabled=bool(os.getenv("VELORA_TEST_MODE", "").lower())
                in ("true", "1"),
                log_env_cpu=True,
                log_env_gpu=True,
                log_env_details=True,
            )
        else:
            # Start new experiment
            self.experiment = Experiment(
                api_key=os.getenv("COMET_API_KEY", None),
                project_name=state.analytics_state.project_name,
                auto_param_logging=False,
                auto_metric_logging=False,
                auto_output_logging=False,
                log_graph=False,
                display_summary_level=0,
                disabled=bool(os.getenv("VELORA_TEST_MODE", "").lower())
                in ("true", "1"),
            )

            self.experiment.set_name(state.analytics_state.experiment_name)
            self.experiment.add_tags(state.analytics_state.tags)

    def config(self) -> Tuple[str, Dict[str, Any]]:
        return self.__class__.__name__, {
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "tags": ",".join(self.tags) if isinstance(self.tags, list) else self.tags,
            "experiment_key": self.experiment_key,
        }

    def info(self) -> str:
        return f"'{self.__class__.__name__}' enabled with 'project_name={self.project_name}', 'experiment_name={self.experiment_name}' and 'tags={self.tags}'."
