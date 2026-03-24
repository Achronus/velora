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

import os
from typing import List

from rich.console import Console, Group
from rich.live import Live

from velora.cli.base.component import (
    Component,
    LiveMetricsCard,
    SetupCard,
    TitleCard,
    TrainingProgressCard,
)


class ConsoleDashboard:
    """
    Orchestrates components into a live console display dashboard.

    Parameters
    ----------
    title : TitleCard
        The title card component
    body : List[Component]
        List of body components (cards, rows, etc.)
    training : ProgressCard (optional)
        Optional training progress card. Displayed after `body`/`setup`. Default is `None`
    live_metrics : LiveMetricsCard (optional)
        Optional live metrics card. Displayed after `body`/`progress`. Default is `None`
    setup : SetupCard (optional)
        Optional setup card. Displayed after `body`. Default is `None`
    width : int (optional)
        Maximum width of the dashboard. Default is `120`
    """

    def __init__(
        self,
        title: TitleCard,
        body: List[Component],
        training: TrainingProgressCard | None = None,
        live_metrics: LiveMetricsCard | None = None,
        setup: SetupCard | None = None,
        width: int = 120,
    ) -> None:
        self.title = title
        self.body = body
        self.training = training
        self.live_metrics = live_metrics
        self.setup = setup

        self._is_setting_up = False
        self._is_training = False

        # Force console display
        self._console_file = os.fdopen(os.dup(1), "w")
        self.console = Console(width=width, file=self._console_file)

        self._live: Live | None = None

    def _check_terminal_width(self) -> None:
        """Block until terminal is wide enough for the dashboard."""
        while True:
            try:
                term_width = os.get_terminal_size().columns
            except OSError:
                return

            if term_width >= self.console.width:
                return

            input(
                f"\nTerminal width ({term_width}) is below the "
                f"recommended minimum ({self.console.width}). "
                "Please resize your terminal and press Enter to continue."
            )

    def _print_static(self) -> None:
        """Print static components (title and body)."""
        self._check_terminal_width()
        self.console.print(self.title.render())

        for component in self.body:
            self.console.print(component.render())

    def _build_setup_display(self) -> Group:
        """Build the setup section display."""
        if self.setup:
            return Group(self.setup.render())

        return Group()

    def _build_training_display(self) -> Group:
        """Build the training section display."""
        components = []
        if self.setup and self.setup._is_complete:
            components.append(self.setup.render())

        if self.training:
            components.append(self.training.render())

        if self.live_metrics:
            components.append(self.live_metrics.render())

        return Group(*components)

    def start_setup(self, total: int | None = None) -> None:
        """
        Start the display in setup mode.

        Parameters
        ----------
        total : int (optional)
            Total number of setup steps (e.g., agents to create).
            If provided, shows determinate progress `x/N`
        """
        self._is_setting_up = True
        self._print_static()

        if self.setup:
            if total is not None:
                self.setup.set_total(total)

            self.setup.start()

        self._live = Live(
            self._build_setup_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def update_setup(self, advance: int = 1) -> None:
        """
        Update setup progress.

        Parameters
        ----------
        advance : int (optional)
            Number of steps to advance. Default is `1`
        """
        if self.setup:
            self.setup.update(advance)

        self._refresh_setup()

    def finish_setup(self) -> None:
        """Mark setup complete."""
        self._is_setting_up = False

        if self.setup:
            self.setup.complete()

        self._refresh_setup()

    def _refresh_setup(self) -> None:
        """Refresh the setup display."""
        if self._live:
            self._live.update(self._build_setup_display())

    def start_training(self) -> None:
        """Start in training mode."""
        self._is_training = True

        if self.training:
            self.training.start()

        if self._live:
            self._live.stop()

        self._live = Live(
            self._build_training_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def finish_training(self) -> None:
        """Mark training complete."""
        self._is_training = False

        if self.training:
            self.training.complete()

        self._refresh_training()

    def _refresh_training(self) -> None:
        """Refresh the training display."""
        if self._live:
            self._live.update(self._build_training_display())

    def update_progress(
        self,
        description: str,
        *,
        advance: int = 1,
        chunk_size: int | None = None,
        env_names: list[str] | None = None,
    ) -> None:
        """
        Update the progress bar.

        Parameters
        ----------
        description : str
            Progress bar to update
        advance : int (optional)
            Advancement progress count. Default is `1`
        chunk_size : int (optional)
            Number of agents in the current chunk. Default is `None`
        env_names : list[str] (optional)
            Environment names in the current chunk. Default is `None`
        """
        if self.training:
            self.training.update(
                description,
                advance=advance,
                chunk_size=chunk_size,
                env_names=env_names,
            )
            self._refresh_training()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()

    def update_losses(self, **kwargs) -> None:
        """
        Update live loss metrics display.

        Parameters
        ----------
        **kwargs : Dict[str, Any]
            Keyword arguments for the loss dataclass
        """
        if self.live_metrics:
            self.live_metrics.update_losses(**kwargs)
            self._refresh_training()

    def update_stats(self, **kwargs) -> None:
        """
        Update live statistic metrics display.

        Parameters
        ----------
        **kwargs : Dict[str, Any]
            Keyword arguments for the stats dataclass
        """
        if self.live_metrics:
            self.live_metrics.update_stats(**kwargs)
            self._refresh_training()
