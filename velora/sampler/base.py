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


class ActionSampler(ABC):
    """
    Base class for all action samplers.

    Samplers are designed to be stateless and only hold configuration,
    never tensors or distributions. Network outputs must flow in as method
    arguments with distributions (where required) are built inside them.
    """

    @abstractmethod
    def deterministic(self, mean: torch.Tensor) -> torch.Tensor:
        """
        Selects the policy's deterministic actions, used during
        evaluation and deployment instead of exploratory sampling.

        Parameters
        ----------
        mean : torch.Tensor
            The action means `(batch_size, out_features)`

        Returns
        -------
        actions : torch.Tensor
            The deterministic actions `(batch_size, out_features)`
        """
        ...


class StochasticSampler(ActionSampler):
    """
    Base class for all stochastic action samplers.
    """

    @abstractmethod
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
        ...

    @abstractmethod
    def evaluate(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Re-evaluates stored actions under the current policy
        distribution during policy updates.

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
        ...


class DeterministicSampler(ActionSampler):
    """
    Base class for all deterministic action samplers.
    """

    @abstractmethod
    def sample(self, mean: torch.Tensor) -> torch.Tensor:
        """
        Selects exploratory actions from the policy's deterministic
        output, typically by applying noise.

        Parameters
        ----------
        mean : torch.Tensor
            The deterministic policy outputs
            `(batch_size, out_features)`

        Returns
        -------
        actions : torch.Tensor
            The exploratory actions `(batch_size, out_features)`
        """
        ...
