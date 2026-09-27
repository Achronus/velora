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
from torch.distributions import Distribution, Independent, Normal

from velora.model.ppo import PPOActor, PPOCritic
from velora.nn.utils import layer_init


class MLP_PPOActor(PPOActor):
    """
    A basic Gaussian MLP actor for PPO with continuous action spaces.
    Has 3 linear layers that use the same `hidden_size`.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    hidden_size : int (optional)
        The number of hidden nodes per layer. Default is `64`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.mean = nn.Sequential(
            layer_init(nn.Linear(in_features, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, out_features), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(out_features))

    def forward(self, obs: torch.Tensor) -> Distribution:
        mean = self.mean(obs)
        return Independent(Normal(mean, self.log_std.exp()), 1)


class MLP_PPOCritic(PPOCritic):
    """
    A basic MLP state-value critic for PPO. Has 3 linear layers
    that use the same `hidden_size`.

    Parameters
    ----------
    in_features : int
        Number of input features
    hidden_size : int (optional)
        The number of hidden nodes per layer. Default is `64`
    """

    def __init__(self, in_features: int, *, hidden_size: int = 64) -> None:
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(in_features, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
