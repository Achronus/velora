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


class ActorCritic(nn.Module, ABC):
    """
    A base class for feedforward actor-critic networks usable by the
    `PPO` and `RPO` trainers.

    Networks are pure function approximators - they output distribution
    parameters and value estimates. Distributions are built by the
    trainer.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the policy's distribution parameters and the critic's
        state-value estimate for a batch of observations.

        Parameters
        ----------
        x : torch.Tensor
            A batch of observations `(batch_size, in_features)`

        Returns
        -------
        action_mean : torch.Tensor
            The action means `(batch_size, out_features)`
        log_std : torch.Tensor
            The action log standard deviations
            `(batch_size, out_features)`
        value : torch.Tensor
            The critic's state-value estimates `(batch_size, 1)`
        """
        ...  # pragma: no cover
