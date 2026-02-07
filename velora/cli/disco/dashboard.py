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

from velora.cli.base.component import (
    CardRow,
    CompileCard,
    Divider,
    LiveMetricsCard,
    LiveMonitoringCard,
    Metric,
    MetricCard,
    ProgressCard,
    TitleCard,
)
from velora.cli.base.constant import Colour
from velora.cli.base.dashboard import ConsoleDashboard
from velora.cli.disco.settings import DiscoDashboardSettings, DiscoLosses, DiscoStats
from velora.utils.format import number_to_short


class DiscoConsoleDashboard(ConsoleDashboard):
    """
    DiscoRL console dashboard.

    Parameters
    ----------
    config : DiscoDashboardSettings
        Configuration settings for the dashboard
    """

    def __init__(self, config: DiscoDashboardSettings) -> None:
        self.config = config
        self.env_categories = config.env_categories
        self.params = config.params

        # Create cards
        title = TitleCard("DiscoRL: Rule Training")
        body = self._body()
        progress = ProgressCard("Meta-training", total=config.meta_steps)
        live_metrics = LiveMetricsCard(DiscoLosses, DiscoStats)
        compile = CompileCard()

        super().__init__(title, body, progress, live_metrics, compile)

    def _body(self) -> List:
        """
        Creates the body of the dashboard.

        Returns
        -------
        body : List[Component]
            List of body components
        """
        return [
            LiveMonitoringCard(self.config.log_dir, self.config.cp_dir),
            CardRow(
                [
                    self._training_step_card(),
                    self._environments_card(),
                    self._disco_card(),
                ]
            ),
        ]

    def _training_step_card(self) -> MetricCard:
        """
        Create a training steps metric card.

        Returns
        -------
        card : MetricCard
            Training steps card
        """
        return MetricCard(
            title="Training Steps",
            metrics=[
                Metric("Steps Per Env", self.config.meta_steps),
                Metric("Agent Updates", self.config.n_updates),
                Metric("Trajectory Size", self.config.seq_len),
                Metric("Batch Size", self.config.batch_size),
            ],
            colour=Colour.PERIWINKLE,
        )

    def _environments_card(self) -> MetricCard:
        """
        Create an environment metric card.

        Returns
        -------
        card : MetricCard
            Environments card
        """
        metrics: List[Metric | Divider] = [
            Metric(cat, count) for cat, count in self.env_categories.items()
        ]
        metrics.append(Divider())
        metrics.append(Metric("Total", self.config.env_total()))

        return MetricCard(
            "Environments",
            metrics,
            Colour.MINT,
        )

    def _disco_card(self) -> MetricCard:
        """
        Create a DiscoRL metric card.

        Returns
        -------
        card : MetricCard
            DiscoRL card
        """
        total_steps = number_to_short(self.config.total_steps())

        return MetricCard(
            title="Agent Stats",
            metrics=[
                Metric("Total Steps", total_steps),
                Divider(),
                Metric("Parameters", "[dim](active/total)[/dim]", separator=""),
                Metric("Policy", str(self.params.policy)),
                Metric("Value", str(self.params.value)),
                Metric("Disco", str(self.params.disco)),
            ],
            colour=Colour.LAVENDER,
        )
