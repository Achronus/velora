# Copyright 2026 Achronus
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


from dataclasses import dataclass, fields
from typing import Dict, Generic, List, Type, TypeVar

import torch


@dataclass
class Stats:
    """Base stats dataclass for type checking."""

    pass


StatsT = TypeVar("StatsT", bound=Stats)


@dataclass
class PPOStats(Stats):
    """
    A set of PPO policy update metrics for a single training iteration,
    averaged over the iteration's mini-batch updates.

    Parameters
    ----------
    policy : float
        The clipped surrogate policy loss
    value : float
        The value function loss
    entropy : float
        The policy distribution's entropy loss
    total : float
        The combined weighted loss
    old_approx_kl : float
        The naive KL divergence approximation `(-log_ratio).mean()`
    approx_kl : float
        The low-variance KL divergence approximation
        `((ratio - 1) - log_ratio).mean()`
    explained_variance : float
        How well the value estimates explain the observed returns.
        `1` is a perfect fit, `0` matches predicting the mean
    clip_frac : float
        The fraction of samples clipped by the policy objective
    """

    policy: float
    value: float
    entropy: float
    total: float
    old_approx_kl: float
    approx_kl: float
    explained_variance: float
    clip_frac: float


class Diagnostics(Generic[StatsT]):
    """
    A base class for accumulating training diagnostic metrics as
    tensors and reducing them into a stats dataclass.

    Metrics are recorded per update as tensors and reduced to floats
    in a single pass, avoiding per-update GPU syncs. Subclasses define
    an explicit `add()` surface for recording and a `summary()` method
    that builds their `Stats` type.

    Parameters
    ----------
    stats_cls : Type[StatsT]
        The stats dataclass the diagnostics reduce into. Its fields
        define the valid metric names
    """

    _metrics: Dict[str, List[torch.Tensor]]

    def __init__(self, stats_cls: Type[StatsT]) -> None:
        self.stats_cls = stats_cls

        self.reset()

    def _record(self, name: str, value: torch.Tensor) -> None:
        """
        Appends a metric value to its accumulator.

        Parameters
        ----------
        name : str
            The metric name. Must match a `stats_cls` field
        value : torch.Tensor
            The metric value for a single update
        """
        names = list(self._metrics.keys())
        if name not in names:
            raise KeyError(
                f"Invalid metric provided: '{name}'. Valid options: '{names}'"
            )

        self._metrics[name].append(value.detach())

    def _means(self) -> Dict[str, float]:
        """
        Reduces the recorded metrics to their means.

        Returns
        -------
        means : Dict[str, float]
            The mean of each recorded metric. Metrics without recorded
            values are excluded
        """
        return {
            name: torch.stack(value).mean().item()
            for name, value in self._metrics.items()
            if value
        }

    def _last(self, name: str) -> float:
        """
        Retrieves the most recently recorded value of a metric.

        Parameters
        ----------
        name : str
            The metric name. Must match a `stats_cls` field

        Returns
        -------
        value : float
            The metric's latest recorded value
        """
        return self._metrics[name][-1].item()

    def reset(self) -> None:
        """Clears all recorded metric values."""
        self._metrics = {f.name: [] for f in fields(self.stats_cls)}


class PPODiagnostics(Diagnostics[PPOStats]):
    """
    Diagnostics for PPO policy updates.

    Records loss and policy ratio metrics after each mini-batch update
    and reduces them into a `PPOStats` at the end of each training
    iteration.

    Parameters
    ----------
    clip_coef : float
        The policy objective's clip coefficient, used to compute the
        clipped sample fraction
    """

    def __init__(self, clip_coef: float) -> None:
        super().__init__(PPOStats)

        self.clip_coef = clip_coef

    @property
    def last_approx_kl(self) -> float:
        """The most recent mini-batch update's KL divergence approximation."""
        return self._last("approx_kl")

    def _variance(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Computes the explained variance of the value estimates.

        Parameters
        ----------
        y_pred : torch.Tensor
            The critic's value estimates `(batch_size,)`
        y_true : torch.Tensor
            The observed returns `(batch_size,)`

        Returns
        -------
        explained_variance : float
            `1` is a perfect fit, `0` matches predicting the mean.
            `nan` when the returns have zero variance
        """
        var_y = torch.var(y_true)
        return float(
            torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y
        )

    def add_losses(
        self,
        policy: torch.Tensor,
        value: torch.Tensor,
        entropy: torch.Tensor,
        total: torch.Tensor,
    ) -> None:
        """
        Records the loss metrics for a single mini-batch update.

        Parameters
        ----------
        policy : torch.Tensor
            The clipped surrogate policy loss
        value : torch.Tensor
            The value function loss
        entropy : torch.Tensor
            The policy distribution's entropy loss
        total : torch.Tensor
            The combined weighted loss
        """
        self._record("policy", policy)
        self._record("value", value)
        self._record("entropy", entropy)
        self._record("total", total)

    @torch.no_grad()
    def add_surrogate(self, log_ratio: torch.Tensor, ratio: torch.Tensor) -> None:
        """
        Records the policy ratio metrics for a single mini-batch
        update.

        Parameters
        ----------
        log_ratio : torch.Tensor
            The log probability ratios between the current and old
            policies `(minibatch_size,)`
        ratio : torch.Tensor
            The probability ratios `exp(log_ratio)`
            `(minibatch_size,)`
        """
        self._record("old_approx_kl", (-log_ratio).mean())
        self._record("approx_kl", ((ratio - 1) - log_ratio).mean())
        self._record("clip_frac", ((ratio - 1.0).abs() > self.clip_coef).float().mean())

    def summary(self, values: torch.Tensor, returns: torch.Tensor) -> PPOStats:
        """
        Reduces the recorded metrics into a complete set of iteration
        stats and empties metric storage.

        Parameters
        ----------
        values : torch.Tensor
            The critic's value estimates `(batch_size,)`
        returns : torch.Tensor
            The observed returns `(batch_size,)`

        Returns
        -------
        stats : PPOStats
            The iteration's diagnostic metrics
        """
        stats_means = self._means()
        variance = self._variance(values, returns)

        self.reset()

        return PPOStats(
            **stats_means,
            explained_variance=variance,
        )
