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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from tensorboardX import SummaryWriter

from velora.tracking.settings import MetricLoggerSettings
from velora.utils.format import create_directory


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
    >>> logger = MetricsLogger("logs", "disco")
    >>> logger.add_writer("meta")
    >>> logger.add_writer("envs/Pong")
    >>> logger.add_writer("envs/Breakout")
    >>> logger.log("meta", step=100, metrics={"loss": 0.5})
    >>> logger.log("envs/Pong", step=100, metrics={"reward": 10.0})
    >>> logger.close()
    """

    def __init__(self, config: MetricLoggerSettings) -> None:
        self.config = config

        self.root_dir = self._create_log_dir(
            self.config.base_dir,
            self.config.experiment_name,
            self.config.timestamp,
        )
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

    @staticmethod
    def _create_log_dir(
        base_dir: str | Path = "logs",
        experiment_name: str = "disco",
        timestamp: bool = True,
    ) -> Path:
        """
        Creates the experiment log directory.

        Parameters
        ----------
        base_dir : str | Path (optional)
            Root directory for all experiments. Default is `logs`
        experiment_name : str (optional)
            Name of experiment. Default is `disco`
        timestamp : bool (optional)
            Whether to append timestamps to experiment directory.
            Uses timestamp format: `ddmmyy_hhmmss`. Default is `True`

        Returns
        -------
        log_dir : Path
            Path to experiment log directory. E.g., `logs/disco_250126_174222/`
        """
        log_dir = create_directory(base_dir, experiment_name, timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

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

    def close(self) -> None:
        """
        Shutdown logger and flush pending writes.

        Waits for all queued metrics to be written before closing
        all TensorBoard writers. Should be called before program exit.
        """
        self.executor.shutdown(wait=True)

        for writer in self.writers.values():
            writer.close()
