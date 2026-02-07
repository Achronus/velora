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

from typing import List, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

from velora.cli.base.component import (
    Component,
    LiveMetricsCard,
    ProgressCard,
    TitleCard,
)
from velora.cli.base.constant import Colour


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
        Optional progress card. Displayed after `live_metrics`. Default is `None`
    live_metrics : LiveMetricsCard (optional)
        Optional live metrics card. Displayed after `body`. Default is `None`
    """

    def __init__(
        self,
        title: TitleCard,
        body: List[Component],
        progress: ProgressCard | None = None,
        live_metrics: LiveMetricsCard | None = None,
    ) -> None:
        self.title = title
        self.body = body
        self.progress = progress
        self.live_metrics = live_metrics

        self.console = Console()
        self._live: Live | None = None

    def _build_display(self) -> Group:
        """Build the full console display."""
        components = [self.title.render()]

        for component in self.body:
            components.append(component.render())  # type: ignore

        if self.live_metrics:
            components.append(self.live_metrics.render())

        if self.progress:
            components.append(self.progress.render())

        return Group(*components)

    def start(self) -> None:
        """Start the live display."""
        if self.progress:
            self.progress.start()

        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def refresh(self) -> None:
        """Refresh the display."""
        if self._live:
            self._live.update(self._build_display())

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

    def print_complete(
        self,
        message: str = "Training complete!",
        details: List[Tuple[str, str]] | None = None,
    ) -> None:
        """
        Print completion message.

        Parameters
        ----------
        message : str
            Completion message
        details : List[Tuple[str, str]] (optional)
            Optional list of `(label, value)` details to show. Default is `None`
        """
        content = f"[bold {Colour.MINT}]✓ {message}[/bold {Colour.MINT}]"

        if details:
            content += "\n"
            for label, value in details:
                content += f"\n{label}: [{Colour.TEAL}]{value}[/{Colour.TEAL}]"

        self.console.print()
        self.console.print(
            Panel(
                content,
                title=f"[bold {Colour.MINT}]Complete[/bold {Colour.MINT}]",
                border_style=Colour.MINT,
                padding=(1, 2),
            )
        )

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
            self.refresh()

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
            self.refresh()
