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


from typing import Tuple

import torch
from torch.distributions import Normal


class PPOActionSampler:
    """
    An action sampler for Gaussian policies.

    Owns the policy's distribution mechanics - building distributions
    from network outputs, sampling actions during rollouts, and
    re-evaluating stored actions during policy updates. Log
    probabilities and entropies are summed over the action dimensions,
    so trainers never touch distribution objects directly.

    Stateless - a single instance can be shared freely.
    """

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

    def sample(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples actions from the policy during rollouts.

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-evaluates stored actions under the current policy during
        policy updates.

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
        dist = self._dist(mean, log_std)
        return dist.log_prob(actions).sum(1), dist.entropy().sum(1)


class RPOActionSampler(PPOActionSampler):
    """
    An action sampler for Robust Policy Optimization (RPO), based on
    [RPO](https://arxiv.org/abs/2212.07536).

    Extends `PPOActionSampler` by perturbing the action mean with
    noise `z ~ Uniform(-alpha, alpha)` when re-evaluating actions
    during policy updates, keeping the policy distribution wide to
    prevent premature entropy collapse. Rollout sampling is untouched.

    Parameters
    ----------
    alpha : float
        The perturbation bound. Higher values increase robustness at
        the cost of slower convergence
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def evaluate(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-evaluates stored actions under the current policy during
        policy updates, perturbing the action mean with
        `z ~ Uniform(-alpha, alpha)` noise.

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
            The log probabilities of the actions under the perturbed
            distribution, summed over the action dimensions
            `(minibatch_size,)`
        entropy : torch.Tensor
            The perturbed distribution's entropy, summed over the
            action dimensions `(minibatch_size,)`
        """
        z = torch.empty_like(mean).uniform_(
            -self.alpha,
            self.alpha,
        )
        return super().evaluate(mean + z, log_std, actions=actions)
