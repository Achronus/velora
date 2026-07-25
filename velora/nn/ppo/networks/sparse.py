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

from typing import Literal

import numpy as np
import torch
from torch import nn

from velora.nn.lnn.blocks import SparseGatedBlock
from velora.nn.lnn.wiring import build_layer_mask
from velora.nn.ppo.networks.base import ActorCritic
from velora.nn.ppo.utils import layer_init
from velora.nn.sparse import SparseLinear


class SparseActor(nn.Module):
    """
    A sparse Multi-Layer Perceptron (MLP) actor for continuous action
    spaces.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    hidden_size : int
        The number of hidden nodes in the trunk
    seed : int
        Random number generator seed for the sparsity masks
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()

        init_std = np.sqrt(2)
        rng = np.random.default_rng(seed)

        self.trunk = nn.Sequential(
            SparseLinear(
                in_features,
                hidden_size,
                build_layer_mask(in_features, hidden_size, rng=rng),
                init_std=init_std,
            ),
            nn.Tanh(),
            SparseLinear(
                hidden_size,
                hidden_size,
                build_layer_mask(hidden_size, hidden_size, rng=rng),
                init_std=init_std,
            ),
            nn.Tanh(),
        )

        self.out = layer_init(nn.Linear(hidden_size, out_features), std=0.01)
        self.logstd = nn.Parameter(torch.zeros(1, out_features))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the policy's distribution parameters for a batch of
        observations.

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
        """
        action_mean: torch.Tensor = self.out(self.trunk(x))
        log_std = self.logstd.expand_as(action_mean)
        return action_mean, log_std


class SparseCritic(nn.Module):
    """
    A sparse Multi-Layer Perceptron (MLP) critic for state-value
    estimation.

    Parameters
    ----------
    in_features : int
        Number of input features
    hidden_size : int
        The number of hidden nodes in the trunk
    seed : int
        Random number generator seed for the sparsity masks
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()

        init_std = np.sqrt(2)
        rng = np.random.default_rng(seed)

        self.trunk = nn.Sequential(
            SparseLinear(
                in_features,
                hidden_size,
                build_layer_mask(in_features, hidden_size, rng=rng),
                init_std=init_std,
            ),
            nn.Tanh(),
            SparseLinear(
                hidden_size,
                hidden_size,
                build_layer_mask(hidden_size, hidden_size, rng=rng),
                init_std=init_std,
            ),
            nn.Tanh(),
        )

        self.out = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the critic's state-value estimate for a batch of
        observations.

        Parameters
        ----------
        x : torch.Tensor
            A batch of observations `(batch_size, in_features)`

        Returns
        -------
        value : torch.Tensor
            The critic's state-value estimates `(batch_size, 1)`
        """
        return self.out(self.trunk(x))


class SparseGatedActor(nn.Module):
    """
    A sparse gated Multi-Layer Perceptron (MLP) actor for continuous action
    spaces.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    hidden_size : int
        The number of hidden nodes in the trunk
    seed : int
        Random number generator seed for the sparsity masks
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()

        init_std = np.sqrt(2)

        self.trunk = nn.Sequential(
            SparseGatedBlock(
                in_features,
                hidden_size,
                seed=seed,
                init_std=init_std,
            ),
            SparseGatedBlock(
                hidden_size,
                hidden_size,
                seed=seed,
                init_std=init_std,
            ),
        )

        self.out = layer_init(nn.Linear(hidden_size, out_features), std=0.01)
        self.logstd = nn.Parameter(torch.zeros(1, out_features))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the policy's distribution parameters for a batch of
        observations.

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
        """
        action_mean: torch.Tensor = self.out(self.trunk(x))
        log_std = self.logstd.expand_as(action_mean)
        return action_mean, log_std


class SparseGatedCritic(nn.Module):
    """
    A sparse gated Multi-Layer Perceptron (MLP) critic for state-value
    estimation.

    Parameters
    ----------
    in_features : int
        Number of input features
    hidden_size : int
        The number of hidden nodes in the trunk
    seed : int
        Random number generator seed for the sparsity masks
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()

        init_std = np.sqrt(2)

        self.trunk = nn.Sequential(
            SparseGatedBlock(
                in_features,
                hidden_size,
                seed=seed,
                init_std=init_std,
            ),
            SparseGatedBlock(
                hidden_size,
                hidden_size,
                seed=seed,
                init_std=init_std,
            ),
        )

        self.out = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the critic's state-value estimate for a batch of
        observations.

        Parameters
        ----------
        x : torch.Tensor
            A batch of observations `(batch_size, in_features)`

        Returns
        -------
        value : torch.Tensor
            The critic's state-value estimates `(batch_size, 1)`
        """
        return self.out(self.trunk(x))


class DualSparseActorCritic(ActorCritic):
    """
    A sparse Multi-Layer Perceptron (MLP) Actor-Critic with options for
    separate `SparseActor` and `SparseCritic` networks.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    method : Literal["linear", "gated"] (optional)
        The type of sparsity layers to use. When `linear` uses `SparseLinear`
        layers. When `gated` uses `SparseGatedBlock` layers. Default is `linear`
    hidden_size : int (optional)
        The number of hidden nodes in the actor and critic trunks.
        Default is `256`
    seed : int (optional)
        Random number generator seed for the sparsity masks. The actor
        uses `seed` and the critic `seed + 1`, keeping their masks
        distinct. Default is `28`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        method: Literal["linear", "gated"] = "linear",
        hidden_size: int = 256,
        seed: int = 28,
    ) -> None:
        super().__init__()

        if method == "linear":
            self.actor = SparseActor(
                in_features,
                out_features,
                hidden_size,
                seed=seed,
            )
            self.critic = SparseCritic(in_features, hidden_size, seed=seed + 1)
        else:
            self.actor = SparseGatedActor(
                in_features,
                out_features,
                hidden_size,
                seed=seed,
            )
            self.critic = SparseGatedCritic(in_features, hidden_size, seed=seed + 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, log_std = self.actor(x)
        return action_mean, log_std, self.critic(x)


class SharedSparseActorCritic(ActorCritic):
    """
    A sparse Multi-Layer Perceptron (MLP) Actor-Critic with a shared trunk.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    method : Literal["linear", "gated"] (optional)
        The type of sparsity layers to use. When `linear` uses `SparseLinear`
        layers. When `gated` uses `SparseGatedBlock` layers. Default is `linear`
    hidden_size : int (optional)
        The number of hidden nodes in the actor and critic trunks.
        Default is `256`
    seed : int (optional)
        Random number generator seed for the sparsity masks. Default is `28`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        method: Literal["linear", "gated"] = "linear",
        hidden_size: int = 256,
        seed: int = 28,
    ) -> None:
        super().__init__()

        init_std = np.sqrt(2)
        rng = np.random.default_rng(seed)

        if method == "linear":
            self.trunk = nn.Sequential(
                SparseLinear(
                    in_features,
                    hidden_size,
                    build_layer_mask(in_features, hidden_size, rng=rng),
                    init_std=init_std,
                ),
                nn.Tanh(),
                SparseLinear(
                    hidden_size,
                    hidden_size,
                    build_layer_mask(hidden_size, hidden_size, rng=rng),
                    init_std=init_std,
                ),
                nn.Tanh(),
            )
        else:
            self.trunk = nn.Sequential(
                SparseGatedBlock(
                    in_features,
                    hidden_size,
                    seed=seed,
                    init_std=init_std,
                ),
                SparseGatedBlock(
                    hidden_size,
                    hidden_size,
                    seed=seed,
                    init_std=init_std,
                ),
            )

        self.actor_mean = layer_init(nn.Linear(hidden_size, out_features), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, out_features))

        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.trunk(x)
        action_mean: torch.Tensor = self.actor_mean(x)
        log_std = self.actor_logstd.expand_as(action_mean)
        return action_mean, log_std, self.critic(x)
