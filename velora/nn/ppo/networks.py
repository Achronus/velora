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
from typing import Tuple

import torch
from torch import nn

from velora.nn.ppo.utils import layer_init


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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class RecurrentActorCritic(nn.Module, ABC):
    """
    A base class for recurrent actor-critic networks usable by the
    `RecurrentPPO` trainer.

    Extends the `ActorCritic` idea with a recurrent state that is
    threaded through calls and reset via `done` flags.
    """

    @abstractmethod
    def init_state(
        self,
        num_envs: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a zeroed recurrent state.

        Parameters
        ----------
        num_envs : int
            The number of parallel environments
        device : torch.device
            Device to load tensors onto

        Returns
        -------
        state : Tuple[torch.Tensor, torch.Tensor]
            The initial recurrent state
        """
        ...  # pragma: no cover

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Computes the policy's distribution parameters and the critic's
        state-value estimate for a batch of observations, threading the
        recurrent state through time.

        Parameters
        ----------
        x : torch.Tensor
            A batch of observations `(batch_size, in_features)`. May
            span multiple timesteps, flattened as
            `(num_steps * num_envs, in_features)`
        state : Tuple[torch.Tensor, torch.Tensor]
            The recurrent state entering the first timestep
        done : torch.Tensor
            The completion flags entering each timestep `(batch_size,)`.
            Resets the recurrent state where set

        Returns
        -------
        action_mean : torch.Tensor
            The action means `(batch_size, out_features)`
        log_std : torch.Tensor
            The action log standard deviations
            `(batch_size, out_features)`
        value : torch.Tensor
            The critic's state-value estimates `(batch_size, 1)`
        new_state : Tuple[torch.Tensor, torch.Tensor]
            The recurrent state after the final timestep
        """
        ...  # pragma: no cover


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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class LSTMActorCritic(RecurrentActorCritic):
    """
    A Long Short-Term Memory (LSTM) Actor-Critic, adapted for
    continuous action spaces from
    [CleanRL - PPO LSTM](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy).

    Uses a shared MLP encoder and LSTM trunk with separate actor and
    critic heads.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    hidden_size : int (optional)
        The number of hidden nodes in the MLP encoder. Default is `64`
    lstm_hidden_size : int (optional)
        The number of hidden nodes in the LSTM. Default is `128`
    lstm_num_layers : int (optional)
        The number of stacked LSTM layers. Default is `1`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_size: int = 64,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(in_features, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, lstm_num_layers)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor_mean = layer_init(
            nn.Linear(lstm_hidden_size, out_features),
            std=0.01,
        )
        self.critic = layer_init(nn.Linear(lstm_hidden_size, 1), std=1.0)
        self.actor_logstd = nn.Parameter(torch.zeros(1, out_features))

    def init_state(
        self,
        num_envs: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a zeroed LSTM state.

        Parameters
        ----------
        num_envs : int
            The number of parallel environments
        device : torch.device
            Device to load tensors onto

        Returns
        -------
        state : Tuple[torch.Tensor, torch.Tensor]
            The initial hidden and cell states, each
            `(lstm_num_layers, num_envs, lstm_hidden_size)`
        """
        shape = (self.lstm.num_layers, num_envs, self.lstm.hidden_size)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )

    def _lstm_forward(
        self,
        hidden: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Steps the LSTM through time, resetting the state where `done`
        is set.

        Parameters
        ----------
        hidden : torch.Tensor
            The encoded observations `(num_steps * num_envs, input_size)`
        state : Tuple[torch.Tensor, torch.Tensor]
            The LSTM state entering the first timestep
        done : torch.Tensor
            The completion flags entering each timestep
            `(num_steps * num_envs,)`

        Returns
        -------
        new_hidden : torch.Tensor
            The LSTM outputs `(num_steps * num_envs, lstm_hidden_size)`
        state : Tuple[torch.Tensor, torch.Tensor]
            The LSTM state after the final timestep
        """
        batch_size = state[0].shape[1]
        hidden = hidden.reshape(-1, batch_size, self.lstm.input_size)
        done = done.float().reshape(-1, batch_size)

        new_hidden = []
        for h, d in zip(hidden, done):
            mask = (1.0 - d).view(1, -1, 1)
            h, state = self.lstm(
                h.unsqueeze(0),
                (mask * state[0], mask * state[1]),
            )
            new_hidden += [h]

        return torch.flatten(torch.cat(new_hidden), 0, 1), state

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Computes the policy's distribution parameters and the critic's
        state-value estimate for a batch of observations, threading the
        LSTM state through time.

        Parameters
        ----------
        x : torch.Tensor
            A batch of observations `(num_steps * num_envs, in_features)`
        state : Tuple[torch.Tensor, torch.Tensor]
            The LSTM state entering the first timestep
        done : torch.Tensor
            The completion flags entering each timestep
            `(num_steps * num_envs,)`

        Returns
        -------
        action_mean : torch.Tensor
            The action means `(num_steps * num_envs, out_features)`
        log_std : torch.Tensor
            The action log standard deviations
            `(num_steps * num_envs, out_features)`
        value : torch.Tensor
            The critic's state-value estimates `(num_steps * num_envs, 1)`
        new_state : Tuple[torch.Tensor, torch.Tensor]
            The LSTM state after the final timestep
        """
        hidden, new_state = self._lstm_forward(self.encoder(x), state, done)
        action_mean: torch.Tensor = self.actor_mean(hidden)
        log_std = self.actor_logstd.expand_as(action_mean)
        return action_mean, log_std, self.critic(hidden), new_state
