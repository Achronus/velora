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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Tuple, Type, TypeVar

from rich.console import Group, RenderableType
from rich.padding import Padding
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
from velora.utils.format import (
    field_to_title,
    format_duration,
    format_path,
    number_to_short,
)

T = TypeVar("T")
T2 = TypeVar("T2")


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
        Separator between label and value. Default is `None`
    """

    label: str
    value: str | int | float
    separator: str = ""

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
    padding : tuple (optional)
        Padding `(top, left, bottom, right)`. Default is `(1, 1, 0, 1)`
    """

    style: str = "dim"
    padding: tuple[int, int, int, int] = (0, 1, 0, 1)

    def render(self) -> Padding:
        """Render the divider as a full-width rule with spacing."""
        return Padding(
            Rule(style=self.style),
            (self.padding[0], self.padding[1], self.padding[2], self.padding[3]),
        )


@dataclass
class Spacer:
    """
    Empty row spacer for vertical alignment in MetricCards.

    Parameters
    ----------
    count : int (optional)
        Number of empty rows. Default is `1`
    """

    count: int = 1

    def render(self) -> Padding:
        """Render empty rows for vertical spacing."""
        rows = "\n" * max(0, self.count - 1)
        padding = (0, 1, 0, 1) if self.count > 0 else (0, 0, 0, 0)
        return Padding(Text(rows), padding)


class Component(ABC):
    """Base for console components."""

    height: int | None = None

    @abstractmethod
    def render(self) -> RenderableType:
        """Render the component."""
        pass

    def get_content_height(self) -> int:
        """
        Calculate the natural height of the component.

        Returns
        -------
        height : int
            Content height in lines. Default is `0`
        """
        return 0


class ProgressComponent(Component, ABC):
    """
    Base for progress-based components with completion states.

    Parameters
    ----------
    colour : str
        Hex colour for border, spinner, and label
    complete_colour : str
        Hex colour for completion state
    """

    def __init__(
        self,
        colour: str,
        complete_colour: str,
    ) -> None:
        self.colour = colour
        self.complete_colour = complete_colour

        self._is_complete = False
        self._elapsed: float | None = None
        self._start_time: float | None = None

        self.progress = self._create_progress()

    def _create_progress(self) -> Progress:
        """
        Create the progress bar.

        Returns
        -------
        bar : Progress
            An auto-updating progress bar
        """
        columns = [
            SpinnerColumn(style=self.colour),
            TextColumn(f"[bold {self.colour}]{{task.description}}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ]
        return Progress(*columns, expand=True)

    def complete(self) -> None:
        """Mark as complete."""
        self._is_complete = True

        if self._start_time is not None:
            self._elapsed = time.perf_counter() - self._start_time

    @abstractmethod
    def start(self) -> None:
        """Start the progress component."""
        raise NotImplementedError()

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the progress component."""
        raise NotImplementedError()

    @abstractmethod
    def _render_complete(self) -> Panel:
        """
        Render the completion state.

        Returns
        -------
        panel : Panel
            Completion panel
        """
        raise NotImplementedError()

    @abstractmethod
    def _render_in_progress(self) -> Panel:
        """
        Render the in progress state.

        Returns
        -------
        panel : Panel
            In progress panel
        """
        raise NotImplementedError()

    def render(self) -> Panel:
        if self._is_complete:
            return self._render_complete()

        return self._render_in_progress()


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
        subtitle_line = f"\n[dim]───[/dim] [bold {self.subtitle_colour}]{self.subtitle}[/bold {self.subtitle_colour}] [dim]───[/dim]"

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
    colour : str (optional)
        Border and command/directory colour. Default is `Colour.AMBER`
    """

    def __init__(
        self,
        log_dir: Path | str,
        checkpoint_dir: Path | str,
        colour: str = Colour.AMBER,
    ) -> None:
        self.log_dir = format_path(log_dir, as_str=True)
        self.checkpoint_dir = format_path(checkpoint_dir, as_str=True)
        self.colour = colour

    def render(self) -> Panel:
        content = Table.grid(padding=(0, 2))
        content.add_column(justify="left")

        content.add_row(
            "[bold white]📊 Tensorboard[/bold white] [dim](run in separate terminal):[/dim]",
        )
        content.add_row(
            f"   [{self.colour}]`tensorboard --logdir={self.log_dir}`[/{self.colour}]",
        )
        content.add_row("")
        content.add_row(
            "[bold white]📁 Checkpoints[/bold white] [dim](saved to):[/dim]",
        )
        content.add_row(
            f"   [{self.colour}]`{self.checkpoint_dir}`[/{self.colour}]",
        )

        return Panel(
            content,
            title=f"[bold {self.colour}]Live Monitoring[/bold {self.colour}]",
            border_style=self.colour,
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
    height : int | None (optional)
        Fixed height for the card. Default is `None` (auto)
    padding : Tuple[int, int, int, int] (optional)
        Padding `(top, right, bottom, left)`.
        Default is `(1, 1, 1, 0)`
    """

    def __init__(
        self,
        title: str,
        metrics: List[Metric | Divider | Spacer],
        colour: str = Colour.LAVENDER,
        height: int | None = None,
        padding: tuple[int, int, int, int] = (1, 1, 1, 0),
    ) -> None:
        self.title = title
        self.metrics = metrics
        self.colour = colour
        self.height = height
        self.padding = padding

    def get_content_height(self) -> int:
        """
        Calculate the natural height of the card.

        Returns
        -------
        height : int
            Content height in lines
        """
        # Panel border (2) + vertical padding + content rows
        top_pad, _, bottom_pad, _ = self.padding
        return 2 + top_pad + bottom_pad + len(self.metrics)

    def render(self) -> Panel:
        from rich.console import Group

        renderables: List[RenderableType] = []
        current_table: Table | None = None

        def _new_table() -> Table:
            t = Table.grid(
                padding=(0, 1, 0, 1),
                expand=True,
                pad_edge=True,
                collapse_padding=False,
            )
            t.add_column(justify="left", style="bold white")
            t.add_column(justify="right", style=self.colour)
            return t

        for metric in self.metrics:
            if isinstance(metric, (Divider, Spacer)):
                if current_table is not None:
                    renderables.append(current_table)
                    current_table = None

                renderables.append(metric.render())
            else:
                if current_table is None:
                    current_table = _new_table()

                current_table.add_row(
                    f"{metric.label}{metric.separator}", metric.format_value()
                )

        if current_table is not None:
            renderables.append(current_table)

        return Panel(
            Group(*renderables),
            title=f"[bold {self.colour}]{self.title}[/bold {self.colour}]",
            border_style=self.colour,
            padding=self.padding,
            height=self.height,
        )


class TrainingProgressCard(ProgressComponent):
    """
    Training progress bar card.

    Parameters
    ----------
    tasks : List[Tuple[str, int]]
        A list of tuples containing `(description, total)`. The first
        task always acts as the main progress bar. An optional second
        task drives the inner chunk sub-component rendered beneath the
        main bar (used by trainers that process trainers in chunks).
        Trainers without chunked progress should pass a single-task list.
    total : int
        Total number of steps
    colour : str (optional)
        Hex colour for border, spinner, and label. Default is `Colour.TEAL`
    complete_colour : str (optional)
        Hex colour for completion state. Default is `Colour.TEAL`
    complete_path : str (optional)
        Optional checkpoint completion path. Default is `None`
    """

    def __init__(
        self,
        tasks: List[Tuple[str, int]],
        total: int,
        colour: str = Colour.TEAL,
        complete_colour: str = Colour.TEAL,
        complete_path: str | None = None,
    ) -> None:
        self.complete_path = (
            format_path(complete_path, as_str=True) if complete_path else None
        )
        self.total = total

        self._tasks = tasks
        self._task_ids: dict[str, TaskID] = {}
        self._current_chunk_size: int | None = None
        self._current_chunk_index: int = 0
        self._current_env_names: List[str] = []

        self._has_inner: bool = len(tasks) > 1
        self._inner_total: int = tasks[1][1] if self._has_inner else 0
        self._inner_count: int = 0

        super().__init__(colour=colour, complete_colour=complete_colour)

    def start(self) -> None:
        self._start_time = time.perf_counter()

        # Only add the first task as a progress bar
        description, total = self._tasks[0]
        task_id = self.progress.add_task(description, total=total)
        self._task_ids[description] = task_id

    def update(
        self,
        description: str,
        *,
        advance: int = 1,
        chunk_size: int | None = None,
        env_names: List[str] | None = None,
    ) -> None:
        """
        Advance the progress card.

        Parameters
        ----------
        description : str
            Task to update
        advance : int (optional)
            Number to increment bar by. Default is `1`
        chunk_size : int (optional)
            Number of agents in the current chunk. Default is `None`
        env_names : List[str] (optional)
            Environment names in the current chunk. Default is `None`
        """
        task_id = self._task_ids.get(description)

        if task_id is not None:
            # First task
            self.progress.update(task_id, advance=advance)
            if self._has_inner:
                self.reset(self._tasks[1][0])

        elif self._has_inner and description == self._tasks[1][0]:
            # Second task
            self._inner_count = min(self._inner_count + advance, self._inner_total)

        if chunk_size is not None:
            self._current_chunk_size = chunk_size
            self._current_chunk_index = self._inner_count
            self._current_env_names = env_names or []

    def reset(self, description: str) -> None:
        """
        Reset part of the progress card.

        Parameters
        ----------
        description : str
            The task to update
        """
        if self._has_inner and description == self._tasks[1][0]:
            self._inner_count = 0
            self._current_chunk_size = None
            self._current_chunk_index = 0
            self._current_env_names = []

    @staticmethod
    def _format_env_names(names: List[str], max_shown: int = 3) -> str:
        """
        Format environment names for display, capping at `max_shown`.

        Parameters
        ----------
        names : List[str]
            Environment names to format
        max_shown : int (optional)
            Maximum number of names to show before summarizing.
            Default is `3`

        Returns
        -------
        formatted : str
            Formatted environment names string
        """
        unique = sorted(set(names))
        if len(unique) <= max_shown:
            return ", ".join(unique)

        return ", ".join(unique[:max_shown]) + f", +{len(unique) - max_shown} more"

    def _render_in_progress(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column()
        table.add_row(self.progress)

        # Update second task
        if self._current_chunk_size is not None:
            n_agents = self._current_chunk_size
            chunk_idx = self._current_chunk_index
            envs_str = self._format_env_names(self._current_env_names)

            table.add_row(
                Text(
                    f"    ↳  Chunk {chunk_idx}/{self._inner_total}"
                    f" | {n_agents} Agent{'s' if n_agents != 1 else ''}"
                    f" | Envs: {envs_str}",
                    style="dim",
                )
            )

        return Panel(
            table,
            title=f"[bold {self.colour}]Training Progress[/bold {self.colour}]",
            border_style=self.colour,
            padding=(1, 2),
        )

    def _render_complete(self) -> Panel:
        total = number_to_short(self.total if self.total else 0)
        elapsed = format_duration(self._elapsed) if self._elapsed else "0s"

        content = Table.grid(padding=(0, 2))
        content.add_column(justify="left")

        if self.complete_path:
            title = f"[bold {self.complete_colour}]✓[/bold {self.complete_colour}] [bold white]Training Complete[/bold white] [dim]({total} meta steps in {elapsed}. Final checkpoint saved to):[/dim]"
        else:
            title = f"[bold {self.complete_colour}]✓ Training Complete[/bold {self.complete_colour}] [dim]({total} meta steps in {elapsed})[/dim]"

        content.add_row(title)

        if self.complete_path:
            content.add_row(
                f"   [{self.colour}]`{self.complete_path}`[/{self.colour}]",
            )

        return Panel(
            content,
            border_style=self.complete_colour,
            padding=(1, 1),
        )


class CardRow(Component):
    """
    A row of 1-3 cards displayed side by side.

    Parameters
    ----------
    cards : List[Component]
        List of cards (1-3) to display in a row
    match_height : bool (optional)
        Whether to match heights of all cards to the tallest. Default is `True`
    """

    def __init__(
        self,
        cards: List[Component],
        match_height: bool = True,
    ) -> None:
        if not 1 <= len(cards) <= 3:
            raise ValueError(f"'cards={len(cards)}' must be 1-3 cards")

        self.cards = cards
        self.match_height = match_height

    def _sync_heights(self) -> None:
        """Set all cards to the height of the tallest card."""
        heights = [card.get_content_height() for card in self.cards]

        if heights:
            max_height = max(heights)

            for card in self.cards:
                card.height = max_height

    def render(self) -> Table:
        if self.match_height:
            self._sync_heights()

        table = Table.grid(expand=True, padding=(0, 1))

        for _ in self.cards:
            table.add_column(ratio=1, vertical="top")

        rendered_cards = [card.render() for card in self.cards]
        table.add_row(*rendered_cards)

        return table


class LiveMetricsCard(Generic[T, T2], Component):
    """
    Two-column live metrics card displaying real-time losses and stats.

    Parameters
    ----------
    losses : Type[T]
        Losses dataclass type
    stats : Type[T2]
        Statistics dataclass type
    colour : str (optional)
        Hex colour for border and values. Default is `Colour.ROSE`
    """

    def __init__(
        self, losses: Type[T], stats: Type[T2], colour: str = Colour.ROSE
    ) -> None:
        self.title = "Live Metrics"
        self.colour = colour
        self.losses: T = losses()
        self.stats: T2 = stats()

    def update_losses(self, **kwargs) -> None:
        """
        Update loss values.

        Parameters
        ----------
        **kwargs : Dict[str, Any]
            Keyword arguments for the loss dataclass
        """
        for key, value in kwargs.items():
            setattr(self.losses, key, value)

    def update_stats(self, **kwargs) -> None:
        """
        Update statistical values.

        Parameters
        ----------
        **kwargs : Dict[str, Any]
            Keyword arguments for the stats dataclass
        """
        for key, value in kwargs.items():
            setattr(self.stats, key, value)

    def _build_side(self, title: str, items: list[tuple[str, float]]) -> Group:
        """Build one side (Losses or Stats) with header, divider, and data."""
        # Header
        header = Text(title, style=f"bold {self.colour}")

        # Data table
        data = Table.grid(expand=True, padding=(0, 2))
        data.add_column(justify="left", style="bold white")
        data.add_column(justify="right", style=self.colour)

        for name, val in items:
            data.add_row(f"{field_to_title(name)}", f"{val:.8f}")

        return Group(header, Rule(style="dim"), data)

    def render(self) -> Panel:
        loss_items = list(vars(self.losses).items())
        stat_items = list(vars(self.stats).items())

        losses_side = self._build_side("Losses", loss_items)
        stats_side = self._build_side("Stats", stat_items)

        # Outer table with vertical divider
        outer = Table.grid(expand=True, padding=(0, 1))
        outer.add_column(ratio=1)
        outer.add_column(width=3, justify="center")
        outer.add_column(ratio=1)

        # Calculate row count for vertical divider
        row_count = (
            max(len(loss_items), len(stat_items)) + 2
        )  # +2 for header and divider
        divider = Text(("│\n" * row_count).strip(), style="dim")

        outer.add_row(losses_side, divider, stats_side)

        return Panel(
            outer,
            title=f"[bold {self.colour}]{self.title}[/bold {self.colour}]",
            border_style=self.colour,
            padding=(1, 1),
        )


class SetupCard(ProgressComponent):
    """
    Setup progress card with spinner that transitions to complete state.

    Covers environment creation, JIT compilation, and other initialization tasks.

    Parameters
    ----------
    colour : str (optional)
        Card progress colour. Default is `Colour.SLATE`
    complete_colour : str (optional)
        Card completion colour. Default is `Colour.SLATE`
    """

    _DESCRIPTION = "Setting up"

    def __init__(
        self,
        colour: str = Colour.SLATE,
        complete_colour: str = Colour.SLATE,
    ) -> None:
        self._total: int | None = None
        self._task_id: TaskID | None = None

        super().__init__(colour=colour, complete_colour=complete_colour)

    def set_total(self, total: int) -> None:
        """
        Set or update the total step count.

        Parameters
        ----------
        total : int
            Total steps for determinate progress
        """
        self._total = total

        if self._task_id is not None:
            self.progress.update(self._task_id, total=total)

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._task_id = self.progress.add_task(self._DESCRIPTION, total=self._total)

    def update(self, advance: int = 1) -> None:
        """
        Advance the progress bar.

        Parameters
        ----------
        advance : int (optional)
            Advancement progress count. Default is `1`
        """
        if self._task_id is not None:
            self.progress.update(self._task_id, advance=advance)

    def _render_in_progress(self) -> Panel:
        return Panel(
            self.progress,
            title=f"[bold {self.colour}]Setup[/bold {self.colour}]",
            border_style=self.colour,
            padding=(1, 2),
        )

    def _render_complete(self) -> Panel:
        elapsed = format_duration(self._elapsed) if self._elapsed else "0s"
        content = f"[bold {self.complete_colour}]✓ Setup Complete[/bold {self.complete_colour}] [dim]({elapsed})[/dim]"

        return Panel(
            content,
            border_style=self.complete_colour,
            padding=(1, 1),
        )
