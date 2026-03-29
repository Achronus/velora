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
from typing import Self

from flax import struct

from velora.utils.format import create_directory


@struct.dataclass(frozen=True)
class RunSettings:
    """
    Dataclass for run directory settings. Co-locates checkpoints, logs,
    and metadata under a single timestamped directory.

    Default settings create the following run directory:
    `runs/trainer_[ddmmyy]_[hhmmss]/`.

    Layout:

        ```text
            runs/trainer_280326_153837/
            ├── checkpoints/       # orbax step dirs + final exported rule
            │   ├── 0/, 100/, ...  # step checkpoints
            │   └── final/         # exported rule
            ├── logs/              # tensorboard writers (meta/, envs/...)
            ├── runtime.log
            └── metadata.json
        ```

    Parameters
    ----------
    name : str (optional)
        Name of the run sub-directory in `base_dir`. Default is `trainer`
    base_dir : Path | str (optional)
        Root directory for run output. Default is `runs`
    freq : int (optional)
        Checkpoint save frequency between timesteps. Default is `100`
    max : int (optional)
        Maximum number of checkpoints to store. Default is `20`
    timestamp : bool (optional)
        Whether to append timestamps to run directory.
        Uses timestamp format: `ddmmyy_hhmmss`. Default is `True`
    _dirpath : Path (optional)
        Cached directory path. Computed automatically on creation.
        Default is `None`
    """

    name: str = "trainer"
    base_dir: Path | str = "runs"
    freq: int = 100
    max: int = 20
    timestamp: bool = True

    _dirpath: Path | None = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        if self._dirpath is None:
            object.__setattr__(
                self,
                "_dirpath",
                create_directory(self.base_dir, self.name, self.timestamp),
            )

    @property
    def dirpath(self) -> Path:
        """
        Get run directory path. Format:
        - `timestamp=True` - `./[base_dir]/[name]_[timestamp]`
        - `timestamp=False` - `./[base_dir]/[name]`
        """
        return self._dirpath  # type: ignore

    @property
    def checkpoint_dir(self) -> Path:
        """
        Get checkpoint subdirectory path.

        Returns
        -------
        path : Path
            `[dirpath]/checkpoints/`
        """
        return self._dirpath / "checkpoints"  # type: ignore

    @property
    def log_dir(self) -> Path:
        """
        Get log subdirectory path for TensorBoard writers.

        Returns
        -------
        path : Path
            `[dirpath]/logs/`
        """
        return self._dirpath / "logs"  # type: ignore

    @classmethod
    def from_path(cls, path: str | Path, **overrides: object) -> Self:
        """
        Create settings that point at an existing run directory.

        Useful when restoring from a checkpoint where the directory
        already exists on disk.

        Parameters
        ----------
        path : str | Path
            Path to an existing run directory
        overrides : kwargs (optional)
            Override any other settings (e.g., `freq`, `max`)

        Returns
        -------
        settings : RunSettings
            Settings with `dirpath` locked to `path`
        """
        resolved = Path(path).resolve()
        return cls(
            base_dir=str(resolved.parent),
            name=resolved.name,
            timestamp=False,
            _dirpath=resolved,
            **overrides,  # type: ignore
        )
