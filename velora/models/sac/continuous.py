from typing import Tuple

import torch
from torch.distributions import Normal

from velora.models.base import LiquidNCPModule, NCPModule


class SACActor(LiquidNCPModule):
    """
    A Liquid NCP Actor Network for the SAC algorithm. Outputs a Gaussian
    distribution over actions.

    Usable with continuous action spaces.
    """

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        *,
        log_std_min: float = -5,
        log_std_max: float = 2,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            num_actions (int): the number of actions
            action_scale (torch.Tensor): scale factor to map normalized actions to
                environment's action range
            action_bias (torch.Tensor): bias/offset to center normalized actions to
                environment's action range
            log_std_min (float, optional): lower bound for the log standard
                deviation of the action distribution. Controls the minimum
                variance of actions
            log_std_max (float, optional): upper bound for the log standard
                deviation of the action distribution. Controls the maximum
                variance of actions
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(num_obs, n_neurons, num_actions * 2, device=device)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    @torch.jit.ignore
    def get_sample(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a set of action samples and log probabilities using the
        reparameterization trick from a Gaussian distribution.

        Parameters:
            mean (torch.Tensor): network prediction means.
            std (torch.Tensor): network standard deviation predictions.

        Returns:
            actions (torch.Tensor): action samples.
            log_probs (torch.Tensor): log probabilities.
        """
        dist = Normal(mean, std)
        x_t = dist.rsample()  # Reparameterization trick
        log_probs = dist.log_prob(x_t)

        return x_t, log_probs

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
        x, new_hidden = self.ncp(obs, hidden)

        mean, _ = torch.chunk(x, 2, dim=-1)

        # Bound actions between [-1, 1]
        actions_normalized = torch.tanh(mean)

        # Scale back to env action space
        actions = actions_normalized * self.action_scale + self.action_bias

        return actions, new_hidden

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the hidden state

        Returns:
            actions (torch.Tensor): the action predictions.
            log_prob (torch.Tensor): log probabilities of actions.
            hidden (torch.Tensor): the new hidden state.
        """
        x, new_hidden = self.ncp(obs, hidden)

        # Split output into mean and log_std
        mean, log_std = torch.chunk(x, 2, dim=-1)

        # Bound between [-20, 2]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # Sample from normal distribution
        x_t, dist_log_probs = self.get_sample(mean, std)
        actions_normalized = torch.tanh(x_t)  # Bounded: [-1, 1]

        # Scale back to environment's action space
        actions = actions_normalized * self.action_scale + self.action_bias

        # Calculate log probability, accounting for tanh
        log_prob = dist_log_probs - torch.log(
            self.action_scale * (1 - actions_normalized.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return actions, log_prob, new_hidden


class SACCritic(LiquidNCPModule):
    """
    A Liquid NCP Critic Network for the SAC algorithm. Estimates Q-values given
    states and actions.

    Usable with continuous action spaces.
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
        super().__init__(num_obs + num_actions, n_neurons, 1, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            actions (torch.Tensor): the batch of actions
            hidden (torch.Tensor, optional): the hidden state

        Returns:
            q_values (torch.Tensor): the Q-Value predictions.
            hidden (torch.Tensor): the new hidden state.
        """
        inputs = torch.cat([obs, actions], dim=-1)
        q_values, new_hidden = self.ncp(inputs, hidden)
        return q_values, new_hidden


class SACCriticNCP(NCPModule):
    """
    An NCP Critic Network for the SAC algorithm. Estimates Q-values given
    states and actions.

    Usable with continuous action spaces.
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
        super().__init__(num_obs + num_actions, n_neurons, 1, device=device)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            actions (torch.Tensor): the batch of actions

        Returns:
            q_values (torch.Tensor): the Q-Value predictions.
        """
        inputs = torch.cat([obs, actions], dim=-1)
        q_values = self.ncp(inputs)
        return q_values
