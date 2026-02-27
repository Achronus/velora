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

import jax

from velora.cli.base.component import (
    CardRow,
    Divider,
    LiveMetricsCard,
    LiveMonitoringCard,
    Metric,
    MetricCard,
    ProgressCard,
    SetupCard,
    Spacer,
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
        progress = ProgressCard(
            "Meta-training",
            total=config.total_steps(),
            complete_path=self.config.complete_path,
        )
        live_metrics = LiveMetricsCard(DiscoLosses, DiscoStats)
        setup = SetupCard()

        super().__init__(title, body, progress, live_metrics, setup)

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
                    self._training_card(),
                    self._environments_card(),
                    self._agent_card(),
                ]
            ),
        ]

    def _training_card(self) -> MetricCard:
        """
        Create a training details metric card.

        Returns
        -------
        card : MetricCard
            Training details card
        """
        compiled = "Yes" if self.config.jit_compile else "No"

        return MetricCard(
            title="Key Details",
            metrics=[
                Metric("Device Type", jax.default_backend().upper()),
                Metric("JIT Compiled", compiled),
                Divider(),
                Metric("Compile Cache", self.config.cache_status),
                Metric("Meta Steps", self.config.meta_steps),
                Metric("Inner Updates", self.config.n_updates),
                Metric("Trajectory Size", self.config.seq_len),
                Metric("Batch Size", self.config.batch_size),
                Divider(),
                Metric("Metric Update Freq (Steps)", self.config.env_total()),
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
        n_rows = 8
        n_categories = len(self.env_categories)

        metrics = [
            *[Metric(cat, count) for cat, count in self.env_categories.items()],
            Spacer(count=n_rows - n_categories),
            Divider(),
            Metric("Total", self.config.env_total()),
        ]

        return MetricCard(
            "Environments",
            metrics,
            Colour.MINT,
        )

    def _agent_card(self) -> MetricCard:
        """
        Create an agent details card unique to DiscoRL.

        Returns
        -------
        card : MetricCard
            Agent details card
        """
        total_steps = number_to_short(self.config.total_steps())

        active_params = number_to_short(
            self.params.policy.active
            + self.params.value.active
            + self.params.disco.active
        )
        total_params = number_to_short(
            self.params.policy.total + self.params.value.total + self.params.disco.total
        )

        return MetricCard(
            title="Agent Details",
            metrics=[
                Metric("Total Training Steps", total_steps),
                Spacer(),
                Divider(),
                Metric("Params per Agent", "[dim](active/total)[/dim]"),
                Metric("Policy", str(self.params.policy)),
                Metric("Value", str(self.params.value)),
                Metric("Disco", str(self.params.disco)),
                Spacer(),
                Divider(),
                Metric("Total Params", f"{active_params}/{total_params}"),
            ],
            colour=Colour.LAVENDER,
        )
