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
from torch import nn

from velora.nn.ppo.networks.base import ActorCritic
from velora.nn.ppo.utils import layer_init


class MLPActorCritic(ActorCritic):
    """
    A Multi-Layer Perceptron (MLP) Actor-Critic from
    [CleanRL - PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy).

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    hidden_size : int (optional)
        The number of hidden nodes in the actor and critic MLPs.
        Default is `64`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_features, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(in_features, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, out_features), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, out_features))

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
        action_mean: torch.Tensor = self.actor_mean(x)
        log_std = self.actor_logstd.expand_as(action_mean)
        return action_mean, log_std, self.critic(x)
