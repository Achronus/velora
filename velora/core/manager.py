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
from typing import Any, Dict, List

import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import FixedIntervalPolicy, LatestN

from velora.config.settings import CheckpointSettings
from velora.config.state import AgentTrainerState


class CheckpointManager:
    """
    Handles checkpoint saving and loading for `AgentTrainerState` using Orbax.

    Supports:
    - Automatic checkpoint rotation (`config.max`)
    - Async saving for non-blocking training
    - Restore from specific step or latest
    - Metadata tracking

    Parameters
    ----------
    env_name : str
        Name of the environment being trained on
    config : CheckpointSettings
        Configuration settings for the manager
    """

    def __init__(
        self,
        env_name: str,
        *,
        config: CheckpointSettings,
    ) -> None:
        self.cp_dir = config.dirpath / env_name
        self.config = config

        self.cp_dir.mkdir(parents=True, exist_ok=True)
        self._last_saved_step = -1

        # Configure checkpointing
        options = ocp.CheckpointManagerOptions(
            save_decision_policy=FixedIntervalPolicy(config.freq),
            preservation_policy=LatestN(config.max),
        )

        self._manager = ocp.CheckpointManager(self.cp_dir, options=options)

    def save(
        self,
        step: int,
        state: AgentTrainerState,
        metrics: Dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """
        Save trainer state to checkpoint.

        Parameters
        ----------
        step : int
            Current training step
        state : AgentTrainerState
            State to save
        metrics : Dict[str, Any] (optional)
            Optional metrics to save alongside state. Default is `None`
        force : bool
            Force save even if within save_interval. Default is `False`

        Returns
        -------
        saved : bool
            Whether the checkpoint was actually saved
        """
        if not force and (step - self._last_saved_step) < self.config.freq:
            return False

        self._manager.save(step, args=ocp.args.PyTreeSave(state))  # type: ignore
        self._last_saved_step = step

        return True

    def restore(
        self,
        step: int | None = None,
        state_template: AgentTrainerState | None = None,
    ) -> AgentTrainerState | None:
        """
        Restore trainer state from checkpoint.

        Parameters
        ----------
        step : int (optional)
            Specific step to restore, or None for latest. Default is `None`
        state_template : AgentTrainerState (optional)
            Template for restoration (helps with structure). Default is `None`

        Returns
        -------
        state : AgentTrainerState | None
            Restored state, or `None` if no checkpoint exists
        """
        if self._manager.latest_step() is None:
            return None

        restore_step = step if step is not None else self._manager.latest_step()

        if state_template is not None:
            restored = self._manager.restore(
                restore_step,
                args=ocp.args.PyTreeRestore(state_template),  # type: ignore
            )
        else:
            restored = self._manager.restore(restore_step)

        return restored  # type: ignore

    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step."""
        return (step - self._last_saved_step) >= self.config.freq

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
