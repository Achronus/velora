from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from velora.models.base import LiquidNCPModule, NCPModule


class SACActorDiscrete(LiquidNCPModule):
    """
    A Liquid NCP Actor Network for the SAC algorithm. Outputs a categorical
    distribution over actions.

    Usable with discrete action spaces.
    """

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            num_actions (int): the number of actions
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(num_obs, n_neurons, num_actions, device=device)

        self.num_actions = num_actions

        self.softmax = nn.Softmax(dim=-1)

    @torch.jit.ignore
    def get_sample(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a set of action samples and log probabilities using a
        Categorical distribution.

        Parameters:
            probs (torch.Tensor): Softmax probabilities for each action

        Returns:
            actions (torch.Tensor): action samples.
            log_probs (torch.Tensor): action log probabilities.
        """
        dist = Categorical(probs=probs)

        actions = dist.sample()
        log_probs = dist.log_prob(actions).unsqueeze(-1)

        return actions, log_probs

    @torch.jit.ignore
    def predict(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a deterministic prediction.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the hidden state

        Returns:
            actions (torch.Tensor): sampled actions
            hidden (torch.Tensor): the new hidden state
        """
        logits, new_hidden = self.ncp(obs, hidden)
        x = self.softmax(logits)
        actions = torch.argmax(x, dim=-1)

        return actions, new_hidden

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the hidden state

        Returns:
            actions (torch.Tensor): the action predictions.
            probs (torch.Tensor): softmax probabilities for each action.
            log_prob (torch.Tensor): log probabilities of actions.
            hidden (torch.Tensor): the new hidden state.
        """
        logits, new_hidden = self.ncp(obs, hidden)
        probs = self.softmax(logits)

        actions, log_prob = self.get_sample(probs)
        return actions, probs, log_prob, new_hidden


class SACCriticDiscrete(LiquidNCPModule):
    """
    A Liquid NCP Critic Network for the SAC algorithm. Estimates Q-values given
    states and actions.

    Usable with discrete action spaces.
    """

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            num_actions (int): the number of actions
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(num_obs, n_neurons, num_actions, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the hidden state

        Returns:
            q_values (torch.Tensor): the Q-Value predictions.
            hidden (torch.Tensor): the new hidden state.
        """
        q_values, new_hidden = self.ncp(obs, hidden)
        return q_values, new_hidden


class SACCriticNCPDiscrete(NCPModule):
    """
    An NCP Critic Network for the SAC algorithm. Estimates Q-values given
    states and actions.

    Usable with discrete action spaces.
    """

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            num_actions (int): the number of actions
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(num_obs, n_neurons, num_actions, device=device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations

        Returns:
            q_values (torch.Tensor): the Q-Value predictions.
        """
        q_values = self.ncp(obs)
        return q_values
