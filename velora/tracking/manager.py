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

import json
from pathlib import Path
from typing import Any, List, Self, Type, TypeVar

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import LatestN

from velora.tracking.metadata import CheckpointMetadata
from velora.tracking.settings import RunSettings

CheckpointState = Any  # Any flax.struct.dataclass instance
M = TypeVar("M", bound=CheckpointMetadata)


class CheckpointManager:
    """
    Handles checkpoint saving and loading for Flax dataclasses using Orbax.

    Supports:
    - Automatic checkpoint rotation (`config.max`)
    - Async saving for non-blocking training
    - Restore from specific step or latest
    - Metadata tracking

    Parameters
    ----------
    config : RunSettings
        Configuration settings for the manager
    """

    def __init__(self, config: RunSettings) -> None:
        self.run_dir = config.dirpath
        self.cp_dir = config.checkpoint_dir
        self.config = config

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cp_dir.mkdir(parents=True, exist_ok=True)

        # Configure checkpointing
        options = ocp.CheckpointManagerOptions(
            preservation_policy=LatestN(config.max),
        )
        self._manager = ocp.CheckpointManager(self.cp_dir, options=options)

    def save(
        self,
        step: int,
        state: CheckpointState,
        *,
        metadata: CheckpointMetadata | None = None,
        force: bool = False,
    ) -> bool:
        """
        Save trainer state to checkpoint.

        Parameters
        ----------
        step : int
            Current training step
        state : CheckpointState
            A Flax dataclass state to save
        metadata : CheckpointMetadata (optional)
            Metadata to persist alongside the checkpoint as
            `metadata.json`. Written on every save, overwriting the
            previous version. Default is `None`
        force : bool (optional)
            Force save regardless of frequency. Default is `False`

        Returns
        -------
        saved : bool
            Whether the checkpoint was actually saved
        """
        if not force and not self.should_save(step):
            return False

        self._manager.save(step, args=ocp.args.PyTreeSave(state), force=force)  # type: ignore

        if metadata is not None:
            meta_path = self.run_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata.to_dict(), indent=2))

        return True

    def restore(
        self,
        step: int | None = None,
        state_template: CheckpointState | None = None,
    ) -> CheckpointState:
        """
        Restore trainer state from checkpoint.

        Parameters
        ----------
        step : int (optional)
            Specific step to restore, or `None` for latest.
            Default is `None`
        state_template : CheckpointState (optional)
            Template for restoration (helps with structure).
            Default is `None`

        Returns
        -------
        state : CheckpointState
            Restored state

        Raises
        ------
        checkpoint_error : ValueError
            If no checkpoint exists in the checkpoint directory
        """
        if self._manager.latest_step() is None:
            raise ValueError(
                f"No checkpoint found in '{self.cp_dir}'. "
                "Ensure the directory contains valid orbax checkpoint files."
            )

        restore_step = step if step is not None else self._manager.latest_step()

        if state_template is not None:
            # Build restore_args with explicit local device sharding
            # to handle cross-device restore (e.g., cuda checkpoint on cpu)
            local_sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
            restore_args = jax.tree.map(
                lambda x: (
                    ocp.type_handlers.ArrayRestoreArgs(sharding=local_sharding)
                    if isinstance(x, jax.ShapeDtypeStruct)
                    else x
                ),
                state_template,
            )

            restored = self._manager.restore(
                restore_step,
                args=ocp.args.PyTreeRestore(
                    state_template,  # type: ignore
                    restore_args=restore_args,
                ),
            )
        else:
            restored = self._manager.restore(restore_step)

        return restored  # type: ignore

    def load_metadata(
        self,
        cls: Type[M],
    ) -> M:
        """
        Load metadata saved alongside checkpoints.

        Parameters
        ----------
        cls : Type[CheckpointMetadata]
            Metadata subclass to deserialize into. Returns a typed
            instance via `cls.from_dict()`

        Returns
        -------
        metadata : CheckpointMetadata
            Typed metadata instance

        Raises
        ------
        file_error : FileNotFoundError
            If `metadata.json` does not exist in the checkpoint directory
        """
        meta_path = self.run_dir / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"No 'metadata.json' found in '{self.run_dir}'. "
                "Ensure the checkpoint was saved with metadata."
            )

        return cls.from_dict(json.loads(meta_path.read_text()))

    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step."""
        return step > 0 and step % self.config.freq == 0

    def wait_until_finished(self) -> None:
        """Wait for any async checkpoint operations to complete."""
        self._manager.wait_until_finished()

    def all_steps(self) -> List[int]:
        """Get all available checkpoint steps."""
        return sorted(self._manager.all_steps())

    @property
    def latest_step(self) -> int | None:
        """Get the latest checkpoint step, or None if no checkpoints exist."""
        return self._manager.latest_step()

    @property
    def directory(self) -> Path:
        """Get the checkpoint directory."""
        return self.cp_dir

    def close(self) -> None:
        """Close the checkpoint manager and release resources."""
        self.wait_until_finished()
        self._manager.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
