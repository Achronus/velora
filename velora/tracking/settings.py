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

from pathlib import Path

from flax import struct

from velora.utils.format import create_directory


@struct.dataclass(frozen=True)
class CheckpointSettings:
    """
    Dataclass for `CheckpointManager` settings.

    Parameters
    ----------
    base_dir : Path | str (optional)
        Directory for saving agent states and checkpoints.
        Default is `checkpoints`
    name : str (optional)
        Name of the checkpoint sub-directory in `base_dir`. Default is `rule_trainer`
    freq : int (optional)
        Checkpoint save frequency between timesteps. Default is `1000`
    max : int (optional)
        Maximum number of checkpoints to store. Default is `20`
    """

    base_dir: Path | str = "checkpoints"
    name: str = "rule_trainer"
    freq: int = 1000
    max: int = 20

    @property
    def dirpath(self) -> Path:
        """Get directory path. Format: `./[base_dir]/[name]`."""
        return Path(self.base_dir, self.name).resolve()


@struct.dataclass(frozen=True)
class MetricLoggerSettings:
    """
    Dataclass for `MetricsLogger` settings.

    Default settings create the following experiment directory:
    `logs/disco_[ddmmyy]_[hhmmss]/`.

    Parameters
    ----------
    base_dir : Path | str (optional)
        Root directory for Tensorboard experiment logs. Default is `logs`
    experiment_name : str (optional)
        Sub-directory name for Tensorboard experiment logs.
        Get's combined with `base_dir`. Default is `disco`.
    timestamp : bool (optional)
        Whether to append timestamps to experiment directory.
        Uses timestamp format: `ddmmyy_hhmmss`. Default is `True`
    """

    base_dir: Path | str = "logs"
    experiment_name: str = "disco"
    timestamp: bool = True

    @property
    def dirpath(self) -> Path:
        """
        Get directory path. Format:
        - `timestamp=True` - `./[base_dir]/[experiment_name]_[timestamp]`
        - `timestamp=False` - `./[base_dir]/[experiment_name]`
        """
        return create_directory(self.base_dir, self.experiment_name, self.timestamp)
