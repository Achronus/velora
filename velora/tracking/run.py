# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Literal

import torch
from tqdm import tqdm

from velora.tracking.logger import MetricsLogger


@dataclass
class RunTrackerConfig:
    """
    Configuration settings for the `RunTracker`. Contains
    details to identify the run.

    Parameters
    ----------
    env_id : str
        The Gymnasium environment ID (e.g., `HalfCheetah-v4`)
    exp_name : str
        The experiment name, shared by all seed runs of the same
        algorithm variant
    project_name : str
        The project name for wandb metric logging
    seed : int
        Random number generator seed used by the run
    """

    env_id: str
    exp_name: str
    project_name: str
    seed: int

    @property
    def group_name(self) -> str:
        """Group name shared by related runs of the same experiment."""
        return f"{self.env_id}_{self.exp_name}"

    @cached_property
    def run_name(self) -> str:
        """
        Name of the run. Timestamped on first access, then fixed for
        the lifetime of the config.
        """
        return f"{self.group_name}_{self.seed}_{int(time.time())}"


class RunTracker:
    """
    A training run tracker that manages training progress with a
    progress bar and metric logging.

    Owns the run's identity (`run_name`), step counter
    (`global_step`), and output directories. Designed for use as a
    context manager - on exit, closes the progress bar, uploads any
    recorded videos, and finishes the logger run.

    Parameters
    ----------
    total_steps : int
        Total number of environment steps used during training.
        Provided to the progress bar for step tracking
    config : RunTrackerConfig
        The run's identity configuration
    metadata : Dict[str, Any] (optional)
        Hyperparameter configuration stored with the run (e.g., the
        agent name and algorithm settings). Default is `None`
    mode : Literal["online", "offline", "disabled"] (optional)
        The wandb logging mode. `online` uploads metrics live,
        `offline` stores them locally for a later `wandb sync`, and
        `disabled` makes metric logging a no-op (useful for tests and
        smoke runs). When `None`, defers to the `WANDB_MODE`
        environment variable. Default is `None`
    """

    def __init__(
        self,
        total_steps: int,
        config: RunTrackerConfig,
        *,
        metadata: Dict[str, Any] | None = None,
        mode: Literal["online", "offline", "disabled"] | None = None,
    ) -> None:
        self.config = config

        self.global_step = 0
        self.run_dir = f"runs/{self.config.exp_name}"
        self.video_dir = Path("runs", "videos", self.config.run_name)

        self.progress = tqdm(
            total=total_steps,
            desc="Training",
            unit="step",
        )

        self.logger = MetricsLogger(
            self.run_dir,
            project=self.config.project_name,
            run_name=self.config.run_name,
            group=self.config.group_name,
            config=metadata,
            mode=mode,
        )

    def __enter__(self) -> "RunTracker":
        """Enters the run context."""
        return self

    def advance(self, n_step: int) -> None:
        """
        Advances the run's `global_step` and progress bar by `n_step`.

        Intended to be called once per environment step with the
        number of transitions gathered (e.g., `num_envs`), keeping
        episodic logs stamped at the exact step they complete.

        Parameters
        ----------
        n_step : int
            The number of environment steps to advance by
        """
        self.global_step += n_step
        self.progress.update(n_step)

    def log(
        self,
        name: str,
        *,
        metrics: Dict[str, float | int],
        bar_details: Dict[str, object],
    ) -> None:
        """
        Logs a set of metrics for a training iteration and updates the
        progress bar's postfix.

        Metrics are logged under the `name` section at the current
        `global_step`. Steps are advanced separately via `advance`.

        Parameters
        ----------
        name : str
            Shared section name for the metrics (e.g., `losses`)
        metrics : Dict[str, float | int]
            Mapping of metric names to scalar values that are logged
        bar_details : Dict[str, object]
            Mapping of stats displayed as the progress bar's postfix
            (e.g., `loss`, `approx_kl`)
        """
        self.logger.log(name, self.global_step, metrics)
        self.progress.set_postfix(bar_details)

    def log_episodic(self, info: Dict) -> None:
        """
        Logs completed episode statistics from a vectorized
        environment step.

        Parses the `final_info` entry populated by
        `RecordEpisodeStatistics` (under `SAME_STEP` autoreset mode),
        logging each finished episode's return and length under the
        `episode` section.

        Uses the current `global_step` without advancing it - steps
        are advanced separately via `advance`.

        Parameters
        ----------
        info : Dict
            The `info` mapping returned by `envs.step`. Episodes are
            detected via the `_episode` mask when `final_info` is
            present; steps without completed episodes are a no-op
        """
        if "final_info" in info:
            episode = info["final_info"]["episode"]
            mask = torch.as_tensor(info["final_info"]["_episode"])

            for idx in mask.nonzero().flatten().tolist():
                self.logger.log(
                    "episode",
                    self.global_step,
                    {
                        "episodic_return": float(episode["r"][idx]),
                        "episodic_length": float(episode["l"][idx]),
                    },
                )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """
        Closes the run.

        Closes the progress bar, uploads any recorded videos to the
        logger's media panel, and finishes the logger run. Exceptions
        are not suppressed.
        """
        self.progress.close()

        videos = sorted(self.video_dir.glob("*.mp4"))
        if videos:
            self.logger.log_video(
                "videos/episodes",
                self.global_step,
                videos,  # type: ignore
            )

        self.logger.close()
