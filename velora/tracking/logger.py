# Copyright 2025 Achronus
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

import logging
import os
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Literal

import wandb


class _TeeWriter:
    """
    File-like object that writes to both a terminal fd and a log file.

    Used to replace `sys.stderr` so that Python-level tracebacks
    and warnings appear on the terminal AND get captured to disk.

    Parameters
    ----------
    terminal_fd : int
        File descriptor of the original terminal stream (saved via
        `os.dup` before fd 2 was redirected)
    log_file : TextIOWrapper
        Open file handle for the log file
    """

    def __init__(self, terminal_fd: int, log_file: TextIOWrapper) -> None:
        self._terminal_fd = terminal_fd
        self._log_file = log_file

    def write(self, msg: str) -> int:
        """Write to both terminal and log file."""
        encoded = msg.encode() if isinstance(msg, str) else msg
        os.write(self._terminal_fd, encoded)

        self._log_file.write(msg)
        self._log_file.flush()
        return len(msg)

    def flush(self) -> None:
        """Flush the log file (terminal fd is unbuffered)."""
        self._log_file.flush()

    def fileno(self) -> int:
        """Return the terminal fd for compatibility."""
        return self._terminal_fd


class RuntimeLogger:
    """
    Captures Python warnings and stderr to a log file.

    Intercepts three streams into a single `runtime.log` file:

    1. **Python warnings** — via `logging.captureWarnings(True)` with
       a `FileHandler`. Catches `warnings.warn()` from JAX, numpy,
       gymnasium, etc.
    2. **Python stderr** — via a tee that writes to both the terminal
       and the log file. Catches tracebacks, logging output, and any
       `print(..., file=sys.stderr)` calls.
    3. **C-level stderr** — by redirecting OS file descriptor 2 to the
       log file. Catches JAX/XLA plugin errors, CUDA warnings, and
       other native code output that bypasses Python's `sys.stderr`.

    Parameters
    ----------
    log_dir : Path
        Directory to write `runtime.log` into (created if needed)
    """

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "runtime.log"
        self._log_file = open(self._log_path, "a", encoding="utf-8")

        # Python warnings → file via logging
        self._handler = logging.FileHandler(self._log_path, encoding="utf-8")
        self._handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self._warnings_logger = logging.getLogger("py.warnings")
        self._warnings_logger.addHandler(self._handler)
        self._warnings_logger.setLevel(logging.WARNING)
        logging.captureWarnings(True)

        # C-level stderr → file only
        # Save the real terminal fd so the tee can still write there
        self._saved_stderr_fd = os.dup(2)
        log_fd = self._log_file.fileno()
        os.dup2(log_fd, 2)  # fd 2 → log file (C-level writes go here)

        # Python stderr → tee (terminal + file)
        self._original_stderr = sys.stderr
        sys.stderr = _TeeWriter(self._saved_stderr_fd, self._log_file)  # type: ignore

    @property
    def path(self) -> Path:
        """Path to the runtime log file."""
        return self._log_path

    def close(self) -> None:
        """Restore stderr and close the log file."""
        # Restore Python stderr
        sys.stderr = self._original_stderr

        # Restore C-level stderr
        os.dup2(self._saved_stderr_fd, 2)
        os.close(self._saved_stderr_fd)

        # Remove logging handler
        logging.captureWarnings(False)
        self._warnings_logger.removeHandler(self._handler)
        self._handler.close()

        # Close log file
        self._log_file.close()


class MetricsLogger:
    """
    A Weights & Biases (wandb) metrics logger.

    Logs metrics to a [W&B](https://wandb.ai) run under `[name]/[key]`
    tags (e.g., `losses/policy`) with `step` as the run's step counter.

    Requires a W&B API key (via `wandb login` or the `WANDB_API_KEY`
    environment variable) when running online. Set `WANDB_MODE=offline`
    to log locally instead; the run is saved in `log_dir` and can be
    uploaded later with `wandb sync`.

    Example root directory format: `runs/trainer_250126_174222/logs/`.

    Parameters
    ----------
    log_dir : Path | str
        Root directory for run output files
    project : str (optional)
        The project name for grouping runs. Default is `velora`
    run_name : str (optional)
        Name of the run. When `None`, wandb generates one automatically.
        Default is `None`
    group : str (optional)
        Group name shared by related runs (e.g., seeds of the same
        experiment). Grouped runs overlay as a mean line with a spread
        band on the dashboard. Default is `None`
    config : dict (optional)
        Hyperparameter configuration stored with the run.
        Default is `None`
    mode : Literal["online", "offline", "disabled"] (optional)
        The wandb logging mode. `online` uploads metrics live,
        `offline` stores them locally for a later `wandb sync`, and
        `disabled` makes all logging a no-op (useful for tests and
        smoke runs). When `None`, defers to the `WANDB_MODE`
        environment variable. Default is `None`

    Examples
    --------
    >>> logger = MetricsLogger("runs/my_run/logs")
    >>> logger.log("losses", step=100, metrics={"policy": 0.5})
    >>> logger.log("charts", step=100, metrics={"episodic_return": 10.0})
    >>> logger.close()
    """

    def __init__(
        self,
        log_dir: Path | str,
        *,
        project: str = "velora",
        run_name: str | None = None,
        group: str | None = None,
        config: dict | None = None,
        mode: Literal["online", "offline", "disabled"] | None = None,
    ) -> None:
        self.root_dir = Path(log_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self._run = wandb.init(
            project=project,
            name=run_name,
            group=group,
            dir=self.root_dir,
            config=config,
            mode=mode,
        )

    def log(self, name: str, step: int, metrics: dict[str, float | int]) -> None:
        """
        Log a set of metrics under a shared name.

        Each metric is logged as `[name]/[key]`, grouping them into a
        single section on the run dashboard.

        Parameters
        ----------
        name : str
            Shared name for the metrics (e.g., `losses`, `episode`)
        step : int
            Training step number
        metrics : dict[str, float | int]
            Mapping of metric names to scalar values
        """
        self._run.log(
            {f"{name}/{k}": float(v) for k, v in metrics.items()},
            step=step,
        )

    def log_video(
        self,
        name: str,
        step: int,
        paths: Path | str | list[Path | str],
    ) -> None:
        """
        Log one or more video files to the run's media panel.

        Multiple videos logged under the same name render as a gallery.

        Parameters
        ----------
        name : str
            Media panel name for the videos (e.g., `videos/episodes`)
        step : int
            Training step number
        paths : Path | str | list[Path | str]
            Path(s) to the video file(s) (e.g., an `mp4`)
        """
        video_paths = paths if isinstance(paths, list) else [paths]
        videos = [wandb.Video(str(path), format="mp4") for path in video_paths]
        self._run.log({name: videos}, step=step)

    def close(self) -> None:
        """
        Shutdown the logger.

        Finishes the wandb run, flushing any pending data. Should be
        called before program exit.
        """
        self._run.finish()
