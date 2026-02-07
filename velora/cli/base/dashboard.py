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

from typing import List

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

from velora.cli.base.component import (
    CompileCard,
    Component,
    LiveMetricsCard,
    ProgressCard,
    TitleCard,
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
    progress : ProgressCard (optional)
        Optional progress card. Displayed after `body`/`compile`. Default is `None`
    live_metrics : LiveMetricsCard (optional)
        Optional live metrics card. Displayed after `body`/`progress`. Default is `None`
    compile : CompileCard (optional)
        Optional compile card. Displayed after `body`. Default is `None`
    width : int (optional)
        Maximum width of the dashboard. Default is `120`
    """

    def __init__(
        self,
        title: TitleCard,
        body: List[Component],
        progress: ProgressCard | None = None,
        live_metrics: LiveMetricsCard | None = None,
        compile: CompileCard | None = None,
        width: int = 120,
    ) -> None:
        self.title = title
        self.body = body
        self.progress = progress
        self.live_metrics = live_metrics
        self.compile = compile

        self._is_compiling = False
        self._is_training = False

        self.console = Console(width=width)
        self._live: Live | None = None

    def _print_static(self) -> None:
        """Print static components (title and body)."""
        self.console.print(self.title.render())

        for component in self.body:
            self.console.print(component.render())

    def _build_compile_display(self) -> Group:
        """Build the compile section display."""
        if self.compile:
            return Group(self.compile.render())

        return Group()

    def _build_training_display(self) -> Group:
        """Build the training section display."""
        components = []

        # If both complete, show them on the same row
        if (
            self.compile
            and self.compile._is_complete
            and self.progress
            and self.progress._is_complete
        ):
            row = Table.grid(expand=True, padding=(0, 1))
            row.add_column(ratio=1)
            row.add_column(ratio=1)
            row.add_row(self.compile.render(), self.progress.render())
            components.append(row)
        else:
            # Show compile complete above progress if applicable
            if self.compile and self.compile._is_complete:
                components.append(self.compile.render())

            if self.progress:
                components.append(self.progress.render())

        if self.live_metrics:
            components.append(self.live_metrics.render())

        return Group(*components)

    def start_compile(self) -> None:
        """Start the display in compile mode."""
        self._is_compiling = True
        self._print_static()

        if self.compile:
            self.compile.start()

        self._live = Live(
            self._build_compile_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def finish_compile(self) -> None:
        """Mark compilation complete."""
        self._is_compiling = False

        if self.compile:
            self.compile.complete()

        self._refresh_compile()

    def _refresh_compile(self) -> None:
        """Refresh the compile display."""
        if self._live:
            self._live.update(self._build_compile_display())

    def start_training(self) -> None:
        """Start in training mode."""
        self._is_training = True

        if self.progress:
            self.progress.start()

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

        if self.progress:
            self.progress.complete()

        self._refresh_training()

    def _refresh_training(self) -> None:
        """Refresh the training display."""
        if self._live:
            self._live.update(self._build_training_display())

    def start(self) -> None:
        """Start the live display (training mode without compile)."""
        self._is_training = True
        self._print_static()

        if self.progress:
            self.progress.start()

        self._live = Live(
            self._build_training_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def update_progress(self, advance: int = 1) -> None:
        """
        Update the progress bar.

        Parameters
        ----------
        advance : int (optional)
            Advancement progress count. Default is `1`
        """
        if self.progress:
            self.progress.update(advance)

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
