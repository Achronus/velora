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

import torch
from torch.distributions import Normal

from velora.sampler.base import StochasticSampler


class GaussianSampler(StochasticSampler):
    """
    An action sampler for Gaussian policies.

    With `alpha > 0`, perturbs the action mean with noise
    `z ~ Uniform(-alpha, alpha)` when re-evaluating actions during
    policy updates, keeping the policy distribution wide to prevent
    premature entropy collapse, based on
    [RPO](https://arxiv.org/abs/2212.07536). Rollout sampling is
    untouched. With `alpha = 0` (the default), behaves as a standard
    Gaussian sampler.

    Parameters
    ----------
    alpha : float (optional)
        The mean perturbation bound applied during `evaluate`. Higher
        values increase robustness at the cost of slower convergence.
        Default is `0.0`
    """

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def _dist(self, mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
        """
        Builds the policy's action distribution.

        Parameters
        ----------
        mean : torch.Tensor
            The action means `(batch_size, out_features)`
        log_std : torch.Tensor
            The action log standard deviations
            `(batch_size, out_features)`

        Returns
        -------
        dist : Normal
            The Gaussian action distribution
        """
        return Normal(mean, log_std.exp())

    def deterministic(self, mean: torch.Tensor) -> torch.Tensor:
        """
        Selects the policy's deterministic actions, used during
        evaluation and deployment instead of exploratory sampling.

        For a Gaussian policy this is the distribution's mode - the
        action mean itself.

        Parameters
        ----------
        mean : torch.Tensor
            The action means `(batch_size, out_features)`

        Returns
        -------
        actions : torch.Tensor
            The deterministic actions `(batch_size, out_features)`
        """
        return mean

    def sample(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples actions from the policy distribution during rollouts.

        Parameters
        ----------
        mean : torch.Tensor
            The action means `(batch_size, out_features)`
        log_std : torch.Tensor
            The action log standard deviations
            `(batch_size, out_features)`

        Returns
        -------
        actions : torch.Tensor
            The sampled actions `(batch_size, out_features)`
        log_probs : torch.Tensor
            The log probabilities of the sampled actions, summed over
            the action dimensions `(batch_size,)`
        """
        dist = self._dist(mean, log_std)
        actions = dist.sample()
        return actions, dist.log_prob(actions).sum(1)

    def evaluate(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Re-evaluates stored actions under the current policy
        distribution during policy updates, perturbing the action
        mean with `z ~ Uniform(-alpha, alpha)` noise when
        `alpha > 0`.

        Parameters
        ----------
        mean : torch.Tensor
            The action means `(minibatch_size, out_features)`
        log_std : torch.Tensor
            The action log standard deviations
            `(minibatch_size, out_features)`
        actions : torch.Tensor
            The rollout actions to re-evaluate
            `(minibatch_size, out_features)`

        Returns
        -------
        log_probs : torch.Tensor
            The log probabilities of the actions, summed over the
            action dimensions `(minibatch_size,)`
        entropy : torch.Tensor
            The policy distribution's entropy, summed over the action
            dimensions `(minibatch_size,)`
        """
        if self.alpha > 0.0:
            z = torch.empty_like(mean).uniform_(-self.alpha, self.alpha)
            mean = mean + z

        dist = self._dist(mean, log_std)
        return dist.log_prob(actions).sum(1), dist.entropy().sum(1)