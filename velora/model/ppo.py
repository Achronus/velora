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

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributions import Distribution


class PPOActor(nn.Module, ABC):
    """
    Base class for PPO actors.

    Subclasses must implement `forward()` for returning the
    policy's action distribution.
    """

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Distribution:
        """
        Computes the policy's action distribution.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of observations `(batch_size, *obs_shape)`

        Returns
        -------
        dist : Distribution
            The action distribution with batch shape `(batch_size,)`
        """
        ...

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Selects actions for a batch of observations during rollouts.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of observations `(batch_size, *obs_shape)`
        deterministic : bool (optional)
            Use the distribution's mode instead of sampling, for
            evaluation. Default is `False`

        Returns
        -------
        actions : torch.Tensor
            The selected actions `(batch_size, *act_shape)`
        log_probs : torch.Tensor
            The log probabilities of the actions `(batch_size, 1)`
        """
        dist: Distribution = self(obs)
        actions = dist.mode if deterministic else dist.sample()
        return actions, dist.log_prob(actions).unsqueeze(-1)

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates stored actions under the current policy.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of observations `(batch_size, *obs_shape)`
        actions : torch.Tensor
            The rollout actions to evaluate `(batch_size, *act_shape)`

        Returns
        -------
        log_probs : torch.Tensor
            The log probabilities of the actions `(batch_size, 1)`
        entropy : torch.Tensor
            The policy distribution's entropy `(batch_size, 1)`
        """
        dist: Distribution = self(obs)

        log_probs = dist.log_prob(actions).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        return log_probs, entropy


class PPOCritic(nn.Module, ABC):
    """
    Base class for PPO critics.

    Subclasses must implement `forward()` for returning
    state-value estimates.
    """

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes state-value estimates.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of observations `(batch_size, *obs_shape)`

        Returns
        -------
        values : torch.Tensor
            The state-value estimates `(batch_size, 1)`
        """
        ...
