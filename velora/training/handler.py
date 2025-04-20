import json
import time
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, List, Literal, Self, Type

from sqlmodel import Session

from velora.utils.format import number_to_short

if TYPE_CHECKING:
    from velora.callbacks import TrainCallback  # pragma: no cover

from velora.gym.wrap import add_core_env_wrappers
from velora.metrics.db import get_db_engine
from velora.models.base import RLModuleAgent
from velora.state import TrainState
from velora.time import ElapsedTime
from velora.training.metrics import TrainMetrics, TrainMetricsBase
from velora.utils.capture import record_episode


class TrainHandlerBase:
    """
    A base class for train handlers.
    """

    def __init__(
        self,
        agent: RLModuleAgent,
        window_size: int,
        callbacks: List["TrainCallback"] | None,
    ) -> None:
        """
        Parameters:
            agent (RLAgent): the agent being trained
            window_size (int): episode window size rate
            callbacks (List[TrainCallback] | None): a list of training callbacks.
                If `None` sets to an empty list
        """
        self.agent = agent
        self.env = self.agent.env
        self.window_size = window_size
        self.callbacks = callbacks or []
        self.device = self.agent.device

        self.state: TrainState | None = None

        self.start_time = 0.0
        self.train_time: ElapsedTime | None = None

        self.engine = get_db_engine()
        self.session: Session | None = None
        self._metrics: TrainMetricsBase | None = None

    def __enter__(self) -> Self:
        """
        Setup the training context, initializing the environment.

        Returns:
            self (Self): the initialized context.
        """
        self.start_time = time.time()

        self.start()
        self.env = add_core_env_wrappers(self.env, self.device)

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

        self.complete()

        self.session.close()

        self.train_time = ElapsedTime.elapsed(self.start_time)

        early_stop_str = "Early stopping target reached!\n" if self.stop() else ""
        print(
            "---------------------------------\n"
            f"{early_stop_str}"
            "Training completed in: "
            f"{number_to_short(self.state.current_ep)} episodes, "
            f"{number_to_short(self._metrics.step_total.item())} steps, "
            f"and {self.train_time}."
        )

    def _run_callbacks(self) -> None:
        """Helper method. Runs the callbacks and updates the training state."""
        for cb in self.callbacks:
            self.state = cb(self.state)

    def complete(self) -> None:
        """Performs `complete` callback event."""
        self.state.status = "complete"
        self._run_callbacks()

    def start(self) -> None:
        """
        Performs `start` callback event.
        """
        self._run_callbacks()

    def stop(self) -> bool:
        """
        Checks if training should be stopped, such as early stopping.

        Returns:
            stop (bool): `True` if training should be stopped, `False` otherwise.
        """
        return self.state.stop_training

    def episode(self, current_ep: int, ep_reward: float) -> None:
        """
        Performs `episode` callback event.

        Parameters:
            current_ep (int): the current training episode index
            ep_reward (float): the episodes reward (return)
        """
        self.state.update(
            status="episode",
            current_ep=current_ep,
            ep_reward=ep_reward,
        )
        self._run_callbacks()

    def record_last_episode(self) -> None:
        """
        If recording videos enabled, captures a recording of the last episode.

        Only works when `TrainCallback.RecordVideos` is applied.

        Filename format: `<env_name>_final-episode-0.mp4`.
        """
        if self.state.record_state is not None:
            dirname = self.state.record_state.dirpath.parent.name
            cp_path = Path("checkpoints", dirname, "videos")
            print()
            record_episode(self.agent, cp_path)

    def save_completed(self) -> None:
        """
        Saves the completed training information to a JSON file in the
        local directory.
        """
        path = Path(self.state.checkpoint_dir, "completed").with_suffix(".json")

        # data to save
        data = {
            "episodes": self.state.current_ep,
            "steps": self._metrics.step_total.item(),
            "time": {
                "hours": self.train_time.hrs,
                "minutes": self.train_time.mins,
                "seconds": round(self.train_time.secs, 4),
            },
            "early_stopped": self.state.stop_training,
            "stats": {
                "reward": {
                    "average": round(self._metrics.reward_moving_avg(), 4),
                    "max": round(self._metrics.reward_moving_max(), 4),
                },
                "loss": {
                    "actor": round(self._metrics._actor_loss.item(), 4),
                    "critic": round(self._metrics._critic_loss.item(), 4),
                    "entropy": round(self._metrics._entropy_loss.item(), 4),
                },
            },
        }

        with path.open("w") as f:
            f.write(json.dumps(data, indent=2))


class TrainHandler(TrainHandlerBase):
    """
    A context manager for handling an agents training state. Compatible with single
    environments.
    """

    def __init__(
        self,
        agent: RLModuleAgent,
        n_episodes: int,
        max_steps: int,
        log_freq: int,
        window_size: int,
        callbacks: List["TrainCallback"] | None,
    ) -> None:
        """
        Parameters:
            agent (RLModuleAgent): the agent being trained
            n_episodes (int): the total number of training episodes
            max_steps (int): maximum number of steps in an episode
            log_freq (int): metric logging frequency (in episodes)
            window_size (int): episode window size rate
            callbacks (List[TrainCallback] | None): a list of training callbacks.
                If `None` sets to an empty list
        """
        super().__init__(agent, window_size, callbacks)

        self.log_freq = log_freq
        self.n_episodes = n_episodes
        self.max_steps = max_steps

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
            self.max_steps,
            device=self.device,
        )
        self._metrics.start_experiment(self.agent.config)

        self.state = TrainState(
            agent=self.agent,
            env=self.env,
            session=self.session,
            total_episodes=self.n_episodes,
            experiment_id=self._metrics.experiment_id,
        )

        return super().__enter__()

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        super().__exit__(exc_type, exc_val, exc_tb)

        if self.state.saving_enabled:
            self.save_completed()

    def start(self) -> None:
        super().start()

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

    def log(self, idx: int, log_type: Literal["episode", "step"]) -> None:
        """
        Performs `logging` callback event.

        Parameters:
            idx (int): the current training step or episode index
            log_type (str): the type of logging method
        """
        if log_type == "episode":
            self.state.update(status="logging", current_ep=idx, logging_type=log_type)
        else:
            self.state.update(status="logging", current_step=idx, logging_type=log_type)

        self._run_callbacks()
