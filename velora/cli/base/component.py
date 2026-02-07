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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from rich.console import RenderableType
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from velora.cli.base.constant import VELORA_LOGO, Colour
from velora.utils.format import number_to_short


@dataclass
class Metric:
    """
    Single metric entry.

    Parameters
    ----------
    label : str
        The name of the metric
    value : str | int | float
        The metric value
    separator : str (optional)
        Separator between label and value. Default is `:`
    """

    label: str
    value: str | int | float
    separator: str = ":"

    def format_value(self) -> str:
        """Format the value for display."""
        if isinstance(self.value, int) and self.value >= 1000:
            return number_to_short(self.value)

        if isinstance(self.value, float):
            if self.value < 0.01:
                return f"{self.value:.0e}"

            return f"{self.value:.2f}"

        return str(self.value)


@dataclass
class Divider:
    """
    Horizontal divider.

    Parameters
    ----------
    style : str (optional)
        Style of the divider. Default is `dim`
    """

    style: str = ""

    def render(self) -> Rule:
        """Render the divider."""
        return Rule(style=self.style)


class Component(ABC):
    """Base for console components."""

    @abstractmethod
    def render(self) -> RenderableType:
        """Render the component."""
        pass


class TitleCard(Component):
    """
    Title card with Velora logo and dynamic subtitle.

    Parameters
    ----------
    subtitle : str
        Subtitle text
    """

    def __init__(self, subtitle: str) -> None:
        self.subtitle = subtitle

        self.subtitle_colour = Colour.LAVENDER
        self.border_colour = Colour.SKY
        self.logo = VELORA_LOGO.format(colour=Colour.SKY)

    def render(self) -> Panel:
        subtitle_line = f"\n[dim]─── [bold {self.subtitle_colour}]{self.subtitle}[/bold {self.subtitle_colour}] ───[/dim]"

        content = Text.from_markup(self.logo + subtitle_line)
        content.justify = "center"

        return Panel(
            content,
            border_style=self.border_colour,
            padding=(1, 2),
        )


class LiveMonitoringCard(Component):
    """
    Card displaying Tensorboard command and checkpoint location.

    Parameters
    ----------
    log_dir : Path | str
        Directory for Tensorboard logs
    checkpoint_dir : Path | str
        Directory for checkpoints
    """

    def __init__(
        self,
        log_dir: Path | str,
        checkpoint_dir: Path | str,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.border_colour = Colour.AMBER

    def render(self) -> Panel:
        content = Table.grid(padding=(0, 2))
        content.add_column(justify="left")

        content.add_row(
            "[bold white]📊 Tensorboard[/bold white] [dim](run in separate terminal):[/dim]",
        )
        content.add_row(
            f"   [{Colour.PERIWINKLE}]`tensorboard --logdir={self.log_dir}`[/{Colour.PERIWINKLE}]",
        )
        content.add_row("")
        content.add_row(
            "[bold white]📁 Checkpoints[/bold white] [dim](saved to):[/dim]",
        )
        content.add_row(
            f"   [{Colour.PERIWINKLE}]`{self.checkpoint_dir}`[/{Colour.PERIWINKLE}]",
        )

        return Panel(
            content,
            title=f"[bold {self.border_colour}]Live Monitoring[/bold {self.border_colour}]",
            border_style=self.border_colour,
            padding=(1, 2),
        )


class MetricCard(Component):
    """
    Customizable metric display card.

    Parameters
    ----------
    title : str
        Card title
    metrics : List[Metric | Separator]
        List of metrics and separators to display
    colour : str (optional)
        Hex colour for border and metric values. Default is `Colour.LAVENDER`
    """

    def __init__(
        self,
        title: str,
        metrics: List[Metric | Divider],
        colour: str = Colour.LAVENDER,
    ) -> None:
        self.title = title
        self.metrics = metrics
        self.colour = colour

    def render(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="bold white")
        table.add_column(justify="left", style=self.colour)

        for metric in self.metrics:
            if isinstance(metric, Divider):
                table.add_row(metric.render())
            else:
                table.add_row(
                    f"{metric.label}{metric.separator}", metric.format_value()
                )

        return Panel(
            table,
            title=f"[bold {self.colour}]{self.title}[/bold {self.colour}]",
            border_style=self.colour,
            padding=(0, 1),
        )


class ProgressCard(Component):
    """
    Training progress bar card.

    Parameters
    ----------
    description : str
        Progress bar caption
    total : int
        Total number of steps
    colour : str
        Hex colour for border, spinner, and label. Default is `Colour.TEAL`
    """

    def __init__(
        self,
        description: str,
        total: int,
        colour: str = Colour.TEAL,
    ) -> None:
        self.description = description
        self.total = total
        self.colour = colour

        self.progress = Progress(
            SpinnerColumn(style=colour),
            TextColumn(f"[bold {colour}]{{task.description}}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        )
        self.task_id: TaskID | None = None

    def start(self) -> None:
        """Start the progress tracker."""
        self.task_id = self.progress.add_task(self.description, total=self.total)

    def update(self, advance: int = 1) -> None:
        """
        Advance the progress bar.

        Parameters
        ----------
        advance : int (optional)
            Advancement progress count. Default is `1`
        """
        if self.task_id is not None:
            self.progress.update(self.task_id, advance=advance)

    def render(self) -> Panel:
        return Panel(
            self.progress,
            title=f"[bold {self.colour}]Training Progress[/bold {self.colour}]",
            border_style=self.colour,
            padding=(1, 2),
        )


class CardRow(Component):
    """
    A row of 1-3 cards displayed side by side.

    Parameters
    ----------
    cards : List[Component]
        List of cards (1-3) to display in a row
    """

    def __init__(self, cards: List[Component]) -> None:
        if not 1 <= len(cards) <= 3:
            raise ValueError(f"'cards={len(cards)}' must be 1-3 cards")

        self.cards = cards

    def render(self) -> Table:
        table = Table.grid(expand=True)

        for _ in self.cards:
            table.add_column(ratio=1)

        rendered_cards = [card.render() for card in self.cards]
        table.add_row(*rendered_cards)

        return table


class LiveMetricsCard(Component):
    """
    Dynamic card displaying real-time training metrics.

    Parameters
    ----------
    title : str (optional)
        Card title. Default is `Live Metrics`
    colour : str (optional)
        Hex colour for border and values. Default is `Colour.TEAL`
    """

    def __init__(
        self,
        title: str = "Live Metrics",
        colour: str = Colour.TEAL,
    ) -> None:
        self.title = title
        self.colour = colour
        self.metrics: Dict[str, float] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update metric values.

        Parameters
        ----------
        metrics : Dict[str, float]
            Mapping of metric names to values
        """
        self.metrics.update(metrics)

    def render(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="bold white")
        table.add_column(justify="left", style=self.colour)

        if not self.metrics:
            table.add_row("[dim]Waiting for data...[/dim]", "")
        else:
            for name, value in self.metrics.items():
                if isinstance(value, float):
                    formatted = f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}"
                else:
                    formatted = str(value)
                table.add_row(f"{name}:", formatted)

        return Panel(
            table,
            title=f"[bold {self.colour}]{self.title}[/bold {self.colour}]",
            border_style=self.colour,
            padding=(0, 1),
        )
