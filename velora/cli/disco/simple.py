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

import time
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from velora.disco.inputs import ActionGroup


class SimpleDashboard:
    """
    Lightweight tqdm-based dashboard for non-interactive or Docker environments.

    Implements the same interface as `ConsoleDashboard` but renders using
    plain tqdm progress bars instead of a Rich live display.

    Parameters
    ----------
    n_steps : int
        Total number of meta-training steps, used as the training bar total
    """

    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self._setup_bar: tqdm | None = None
        self._train_bar: tqdm | None = None
        self._postfix: dict = {}
        self._step_start: float | None = None

    def start_setup(self, total: int | None = None) -> None:
        """Start a tqdm progress bar for the setup phase."""
        self._setup_bar = tqdm(
            total=total,
            desc="Setting up",
            unit="step",
            leave=True,
            dynamic_ncols=True,
        )

    def update_setup(self, advance: int = 1) -> None:
        """Advance the setup progress bar."""
        if self._setup_bar:
            self._setup_bar.update(advance)

    def finish_setup(self) -> None:
        """Close the setup bar and print a completion message."""
        if self._setup_bar:
            self._setup_bar.set_description("Setup complete")
            self._setup_bar.close()
            self._setup_bar = None

    def start_training(self) -> None:
        """Start a tqdm progress bar for the training phase."""
        self._train_bar = tqdm(
            total=self._n_steps,
            desc="Training",
            unit="step",
            leave=True,
            dynamic_ncols=True,
        )

    def update_progress(
        self,
        description: str,
        *,
        advance: int = 1,
        group: "ActionGroup | None" = None,
    ) -> None:
        """
        Update training progress.

        ``"Meta Steps"`` advances the training bar by one step.
        ``"Inner Updates"`` sets the current group label as a postfix.
        """
        if not self._train_bar:
            return

        if description == "Meta Steps":
            if self._step_start is not None:
                elapsed = time.perf_counter() - self._step_start
                self._postfix["step"] = f"{elapsed:.1f}s"
                self._step_start = None
            if self._postfix:
                self._train_bar.set_postfix(self._postfix, refresh=False)
            self._train_bar.update(advance)
        elif description == "Inner Updates" and group is not None:
            if self._step_start is None:
                self._step_start = time.perf_counter()

            n_envs = len(group.indices)
            label = f"{group.n_actions} Action Group | {n_envs} Environment{'s' if n_envs != 1 else ''}"
            self._train_bar.set_postfix_str(label, refresh=False)

    def update_stats(self, **kwargs) -> None:
        """Accumulate reward stats to display as postfix on the next meta-step tick."""
        avg_reward = kwargs.get("avg_reward")
        if avg_reward is not None:
            self._postfix["reward"] = f"{avg_reward:.3f}"

    def update_losses(self, **kwargs) -> None:
        """Accumulate loss metrics to display as postfix on the next meta-step tick."""
        meta = kwargs.get("meta")
        grad_norm = kwargs.get("gradient_norm")
        if meta is not None:
            self._postfix["loss"] = f"{meta:.3f}"
        if grad_norm is not None:
            self._postfix["grad"] = f"{grad_norm:.3f}"

    def finish_training(self) -> None:
        """Close the training bar."""
        if self._train_bar:
            self._train_bar.close()
            self._train_bar = None
