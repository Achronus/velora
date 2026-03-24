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
from concurrent.futures import ThreadPoolExecutor
from io import TextIOWrapper
from pathlib import Path
from typing import Dict

from tensorboardX import SummaryWriter

from velora.tracking.settings import MetricLoggerSettings


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
    Asynchronous TensorBoard metrics logger.

    Writes metrics to TensorBoard in a background thread to avoid blocking
    the training loop. Uses a single-worker thread pool to ensure writes
    are sequential and non-blocking.

    Example root directory format: `logs/disco_250126_174222/`.

    Parameters
    ----------
    config : MetricLoggerSettings
        Configuration for the metric logger

    Examples
    --------
    >>> logger = MetricsLogger(MetricLoggerSettings())
    >>> logger.add_writer("meta")
    >>> logger.add_writer("envs/Pong")
    >>> logger.add_writer("envs/Breakout")
    >>> logger.log("meta", step=100, metrics={"loss": 0.5})
    >>> logger.log("envs/Pong", step=100, metrics={"reward": 10.0})
    >>> logger.close()
    """

    def __init__(self, config: MetricLoggerSettings) -> None:
        self.config = config

        self.root_dir = self.config.dirpath
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.writers: Dict[str, SummaryWriter] = {}
        self.executor = ThreadPoolExecutor(max_workers=1)

    def add_writer(self, name: str) -> None:
        """
        Add a new TensorBoard writer for a specific component.

        Creates a subdirectory under the root log directory and
        initializes a `SummaryWriter` for it.

        Parameters
        ----------
        name : str
            Writer name, used as subdirectory (e.g., `meta`, `envs/Pong`)
        """
        if name in self.writers:
            return

        writer_dir = self.root_dir / name
        writer_dir.mkdir(parents=True, exist_ok=True)
        self.writers[name] = SummaryWriter(str(writer_dir))

    def log(self, writer_name: str, step: int, metrics: dict) -> None:
        """
        Log metrics asynchronously to a specific writer.

        Submits metrics to be written in a background thread. Returns
        immediately without waiting for the write to complete.

        Parameters
        ----------
        writer_name : str
            Name of writer to use (must be added via `add_writer` first)
        step : int
            Training step number
        metrics : dict
            Mapping of metric names to scalar values

        Raises
        ------
        KeyError
            If writer_name hasn't been added
        """
        if writer_name not in self.writers:
            raise KeyError(
                f"Writer '{writer_name}' not found. Call 'add_writer()' first."
            )

        self.executor.submit(self._write, writer_name, step, metrics)

    def _write(self, writer_name: str, step: int, metrics: dict) -> None:
        """
        Write metrics to TensorBoard.

        Called in background thread by the executor.

        Parameters
        ----------
        writer_name : str
            Name of writer to use
        step : int
            Training step number
        metrics : dict
            Mapping of metric names to scalar values
        """
        writer = self.writers[writer_name]

        for k, v in metrics.items():
            writer.add_scalar(k, float(v), step)

        self.flush()

    def close(self) -> None:
        """
        Shutdown logger and flush pending writes.

        Waits for all queued metrics to be written before closing
        all TensorBoard writers. Should be called before program exit.
        """
        self.executor.shutdown(wait=True)

        for writer in self.writers.values():
            writer.close()

    def flush(self) -> None:
        """Flush all writers to disk for live TensorBoard monitoring."""
        for writer in self.writers.values():
            writer.flush()
