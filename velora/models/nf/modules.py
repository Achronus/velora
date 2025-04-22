from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.optim as optim

from velora.models.config import CriticConfig, EntropyParameters
from velora.models.sac.continuous import SACActor, SACCriticNCP
from velora.models.sac.discrete import SACActorDiscrete, SACCriticNCPDiscrete
from velora.utils.torch import soft_update


class BaseModule(ABC):
    """
    A base module for all agent modules.
    """

    @abstractmethod
    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Creates a state dictionary for the module.

        Returns:
            state_dict (Dict[str, Dict[str, Any]]): the state dicts for the module, including networks and optimizers.
        """
        pass  # pragma: no cover

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Load the modules state dict from a previously saved state.

        Parameters:
            state_dict (Dict[str, Dict[str, Any]]): a previously saved state dict
        """
        pass  # pragma: no cover


class ActorModule(BaseModule):
    """
    An Actor module for NeuroFlow. Uses a Liquid NCP SAC Actor with a
    Gaussian policy.

    Usable with continuous action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        n_neurons: int,
        action_dim: int,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        *,
        log_std_min: float = -5,
        log_std_max: float = 2,
        optim: Type[optim.Optimizer] = optim.Adam,
        lr: float = 3e-4,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            state_dim (int): dimension of the state space
            n_neurons (int): number of decision/hidden neurons
            action_dim (int): dimension of the action space
            action_scale (torch.Tensor): scale factor to map normalized actions to
                environment's action range
            action_bias (torch.Tensor): bias/offset to center normalized actions to
                environment's action range
            log_std_min (float, optional): minimum log standard deviation
            log_std_max (float, optional): maximum log standard deviation
            optim (Type[optim.Optimizer], optional): a `PyTorch` optimizer class
            lr (float, optional): optimizer learning rate
            device (torch.device, optional): the device to perform computations on
        """
        self.state_dim = state_dim
        self.n_neurons = n_neurons
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.log_std = (log_std_min, log_std_max)
        self.lr = lr
        self.device = device

        self.network = SACActor(
            state_dim,
            n_neurons,
            action_dim,
            action_scale,
            action_bias,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            device=device,
        ).to(device)

        self.hidden_size = self.network.ncp.hidden_size

        self.optim = optim(self.network.parameters(), lr=lr)

        self.config = self.network.config()

        self.active_params = self.config.active_params
        self.total_params = self.config.total_params

        self.network: SACActor = torch.jit.script(self.network)

    def gradient_step(self, loss: torch.Tensor) -> None:
        """
        Performs a gradient update step.

        Parameters:
            loss (torch.Tensor): network loss
        """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def predict(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Makes a deterministic prediction using the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the current hidden state

        Returns:
            actions (torch.Tensor): the action predictions.
            hidden (torch.Tensor): the new hidden state.
        """
        action, hidden = self.network.predict(obs, hidden)
        return action, hidden

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the current hidden state

        Returns:
            actions (torch.Tensor): the action predictions.
            log_prob (torch.Tensor): log probabilities of actions.
            hidden (torch.Tensor): the new hidden state.
        """
        action, log_prob, hidden = self.network(obs, hidden)
        return action, log_prob, hidden

    def eval_mode(self) -> None:
        """Sets the network to evaluation mode."""
        self.network.eval()

    def train_mode(self) -> None:
        """Sets the network to training mode."""
        self.network.train()

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "actor": self.network.state_dict(),
            "actor_optim": self.optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        self.network.load_state_dict(state_dict["actor"])
        self.optim.load_state_dict(state_dict["actor_optim"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"n_neurons={self.n_neurons}, "
            f"action_dim={self.action_dim}, "
            f"action_scale={self.action_scale}, "
            f"action_bias={self.action_bias}, "
            f"optim={type(self.optim).__name__}, "
            f"log_std={self.log_std}, "
            f"lr={self.lr}, "
            f"device={self.device})"
        )


class ActorModuleDiscrete(BaseModule):
    """
    An Actor module for NeuroFlow. Uses a Liquid NCP SAC Actor with a
    Categorical policy.

    Usable with discrete action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        n_neurons: int,
        action_dim: int,
        *,
        optim: Type[optim.Optimizer] = optim.Adam,
        lr: float = 3e-4,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            state_dim (int): dimension of the state space
            n_neurons (int): number of decision/hidden neurons
            action_dim (int): dimension of the action space
            optim (Type[optim.Optimizer], optional): a `PyTorch` optimizer class
            lr (float, optional): optimizer learning rate
            device (torch.device, optional): the device to perform computations on
        """
        self.state_dim = state_dim
        self.n_neurons = n_neurons
        self.action_dim = action_dim
        self.lr = lr
        self.device = device

        self.network = SACActorDiscrete(
            state_dim,
            n_neurons,
            action_dim,
            device=device,
        ).to(device)

        self.hidden_size = self.network.ncp.hidden_size

        self.optim = optim(self.network.parameters(), lr=lr)

        self.config = self.network.config()

        self.active_params = self.config.active_params
        self.total_params = self.config.total_params

        self.network: SACActorDiscrete = torch.jit.script(self.network)

    def gradient_step(self, loss: torch.Tensor) -> None:
        """
        Performs a gradient update step.

        Parameters:
            loss (torch.Tensor): network loss
        """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def predict(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Makes a deterministic prediction using the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the current hidden state

        Returns:
            actions (torch.Tensor): the action predictions.
            hidden (torch.Tensor): the new hidden state.
        """
        action, hidden = self.network.predict(obs, hidden)
        return action, hidden

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the current hidden state

        Returns:
            actions (torch.Tensor): the action predictions.
            probs (torch.Tensor): softmax probabilities for each action.
            log_prob (torch.Tensor): log probabilities of actions.
            hidden (torch.Tensor): the new hidden state.
        """
        actions, probs, log_prob, hidden = self.network(obs, hidden)
        return actions, probs, log_prob, hidden

    def eval_mode(self) -> None:
        """Sets the network to evaluation mode."""
        self.network.eval()

    def train_mode(self) -> None:
        """Sets the network to training mode."""
        self.network.train()

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "actor": self.network.state_dict(),
            "actor_optim": self.optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        self.network.load_state_dict(state_dict["actor"])
        self.optim.load_state_dict(state_dict["actor_optim"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"n_neurons={self.n_neurons}, "
            f"action_dim={self.action_dim}, "
            f"optim={type(self.optim).__name__}, "
            f"lr={self.lr}, "
            f"device={self.device})"
        )


class CriticModule(BaseModule):
    """
    A Critic module for NeuroFlow. Uses a pair of NCP SAC Critic's with separate
    target networks to estimate Q-values.

    Usable with continuous action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        n_neurons: int,
        action_dim: int,
        *,
        optim: Type[optim.Optimizer] = optim.Adam,
        lr: float = 3e-4,
        tau: float = 0.005,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            state_dim (int): dimension of the state space
            n_neurons (int): number of decision/hidden neurons
            action_dim (int): dimension of the action space
            optim (Type[optim.Optimizer], optional): a `PyTorch` optimizer class
            lr (float, optional): optimizer learning rates
            tau (float, optional): soft target network update factor
            device (torch.device, optional): the device to perform computations on
        """
        self.state_dim = state_dim
        self.n_neurons = n_neurons
        self.action_dim = action_dim
        self.lr = lr
        self.tau = tau
        self.device = device

        self.network1 = SACCriticNCP(
            state_dim,
            n_neurons,
            action_dim,
            device=device,
        ).to(device)

        self.network2 = SACCriticNCP(
            state_dim,
            n_neurons,
            action_dim,
            device=device,
        ).to(device)

        self.target1 = deepcopy(self.network1)
        self.target2 = deepcopy(self.network2)

        self.optim1 = optim(self.network1.parameters(), lr=lr)
        self.optim2 = optim(self.network2.parameters(), lr=lr)

        self.config = CriticConfig(
            critic1=self.network1.config(),
            critic2=self.network2.config(),
        )

        self.active_params = (
            self.config.critic1.active_params + self.config.critic2.active_params
        )
        self.total_params = (
            self.config.critic1.total_params + self.config.critic2.total_params
        )

        self.network1: SACCriticNCP = torch.jit.script(self.network1)
        self.network2: SACCriticNCP = torch.jit.script(self.network2)

        self.target1: SACCriticNCP = torch.jit.script(self.target1)
        self.target2: SACCriticNCP = torch.jit.script(self.target2)

    def update_targets(self) -> None:
        """
        Performs a soft update on the target networks.
        """
        soft_update(self.network1, self.target1, tau=self.tau)
        soft_update(self.network2, self.target2, tau=self.tau)

    def gradient_step(self, c1_loss: torch.Tensor, c2_loss: torch.Tensor) -> None:
        """
        Performs a gradient update step.

        Parameters:
            c1_loss (torch.Tensor): critic loss for first network
            c2_loss (torch.Tensor): critic loss for second network
        """
        self.optim1.zero_grad()
        c1_loss.backward()
        self.optim1.step()

        self.optim2.zero_grad()
        c2_loss.backward()
        self.optim2.step()

    def predict(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Makes a prediction using the critic networks.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            actions (torch.Tensor): the batch of actions

        Returns:
            q_values1 (torch.Tensor): the Q-Value predictions from the first network.
            q_values2 (torch.Tensor): the Q-Value predictions from the second network.
        """
        q_values1 = self.network1(obs, actions)
        q_values2 = self.network2(obs, actions)

        return q_values1, q_values2

    def target_predict(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Makes a prediction using the target networks.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            actions (torch.Tensor): the batch of actions

        Returns:
            next_q (torch.Tensor): the smallest Q-Value predictions between the target networks.
        """
        q_values1 = self.target1(obs, actions)
        q_values2 = self.target2(obs, actions)

        return torch.min(q_values1, q_values2)

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "critic": self.network1.state_dict(),
            "critic2": self.network2.state_dict(),
            "critic_target": self.target1.state_dict(),
            "critic2_target": self.target2.state_dict(),
            "critic_optim": self.optim1.state_dict(),
            "critic2_optim": self.optim2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        self.network1.load_state_dict(state_dict["critic"])
        self.network2.load_state_dict(state_dict["critic2"])

        self.target1.load_state_dict(state_dict["critic_target"])
        self.target2.load_state_dict(state_dict["critic2_target"])

        self.optim1.load_state_dict(state_dict["critic_optim"])
        self.optim2.load_state_dict(state_dict["critic2_optim"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"n_neurons={self.n_neurons}, "
            f"action_dim={self.action_dim}, "
            f"optim={type(self.optim1).__name__}, "
            f"lr={self.lr}, "
            f"tau={self.tau}, "
            f"device={self.device})"
        )


class CriticModuleDiscrete(BaseModule):
    """
    A Critic module for NeuroFlow. Uses a pair of NCP SAC Critic's with separate
    target networks to estimate Q-values.

    Usable with discrete action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        n_neurons: int,
        action_dim: int,
        *,
        optim: Type[optim.Optimizer] = optim.Adam,
        lr: float = 3e-4,
        tau: float = 0.005,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            state_dim (int): dimension of the state space
            n_neurons (int): number of decision/hidden neurons
            action_dim (int): dimension of the action space
            optim (Type[optim.Optimizer], optional): a `PyTorch` optimizer class
            lr (float, optional): optimizer learning rates
            tau (float, optional): soft target network update factor
            device (torch.device, optional): the device to perform computations on
        """
        self.state_dim = state_dim
        self.n_neurons = n_neurons
        self.action_dim = action_dim
        self.lr = lr
        self.tau = tau
        self.device = device

        self.network1 = SACCriticNCPDiscrete(
            state_dim,
            n_neurons,
            action_dim,
            device=device,
        ).to(device)

        self.network2 = SACCriticNCPDiscrete(
            state_dim,
            n_neurons,
            action_dim,
            device=device,
        ).to(device)

        self.target1 = deepcopy(self.network1)
        self.target2 = deepcopy(self.network2)

        self.optim1 = optim(self.network1.parameters(), lr=lr)
        self.optim2 = optim(self.network2.parameters(), lr=lr)

        self.config = CriticConfig(
            critic1=self.network1.config(),
            critic2=self.network2.config(),
        )

        self.active_params = (
            self.config.critic1.active_params + self.config.critic2.active_params
        )
        self.total_params = (
            self.config.critic1.total_params + self.config.critic2.total_params
        )

        self.network1: SACCriticNCPDiscrete = torch.jit.script(self.network1)
        self.network2: SACCriticNCPDiscrete = torch.jit.script(self.network2)

        self.target1: SACCriticNCPDiscrete = torch.jit.script(self.target1)
        self.target2: SACCriticNCPDiscrete = torch.jit.script(self.target2)

    def update_targets(self) -> None:
        """
        Performs a soft update on the target networks.
        """
        soft_update(self.network1, self.target1, tau=self.tau)
        soft_update(self.network2, self.target2, tau=self.tau)

    def gradient_step(self, c1_loss: torch.Tensor, c2_loss: torch.Tensor) -> None:
        """
        Performs a gradient update step.

        Parameters:
            c1_loss (torch.Tensor): critic loss for first network
            c2_loss (torch.Tensor): critic loss for second network
        """
        self.optim1.zero_grad()
        c1_loss.backward()
        self.optim1.step()

        self.optim2.zero_grad()
        c2_loss.backward()
        self.optim2.step()

    def predict(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Makes a prediction using the critic networks.

        Parameters:
            obs (torch.Tensor): the batch of state observations

        Returns:
            q_values1 (torch.Tensor): the Q-Value predictions from the first network.
            q_values2 (torch.Tensor): the Q-Value predictions from the second network.
        """
        q_values1 = self.network1(obs)
        q_values2 = self.network2(obs)

        return q_values1, q_values2

    def target_predict(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Makes a prediction using the target networks.

        Parameters:
            obs (torch.Tensor): the batch of state observations

        Returns:
            next_q (torch.Tensor): the smallest Q-Value predictions between the target networks.
        """
        q_values1 = self.target1(obs)
        q_values2 = self.target2(obs)

        return torch.min(q_values1, q_values2)

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "critic": self.network1.state_dict(),
            "critic2": self.network2.state_dict(),
            "critic_target": self.target1.state_dict(),
            "critic2_target": self.target2.state_dict(),
            "critic_optim": self.optim1.state_dict(),
            "critic2_optim": self.optim2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        self.network1.load_state_dict(state_dict["critic"])
        self.network2.load_state_dict(state_dict["critic2"])

        self.target1.load_state_dict(state_dict["critic_target"])
        self.target2.load_state_dict(state_dict["critic2_target"])

        self.optim1.load_state_dict(state_dict["critic_optim"])
        self.optim2.load_state_dict(state_dict["critic2_optim"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"n_neurons={self.n_neurons}, "
            f"action_dim={self.action_dim}, "
            f"optim={type(self.optim1).__name__}, "
            f"lr={self.lr}, "
            f"tau={self.tau}, "
            f"device={self.device})"
        )


class EntropyModule(BaseModule):
    """
    An Entropy module for NeuroFlow. Uses automatic entropy tuning from SAC
    based on the paper: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905).

    Usable with continuous action spaces.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        initial_alpha: float = 1.0,
        optim: Type[optim.Optimizer] = optim.Adam,
        lr: float = 3e-4,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            action_dim (int): dimension of the action space
            initial_alpha (float, optional): the starting entropy coefficient value
            optim (Type[optim.Optimizer], optional): a `PyTorch` optimizer class
            lr (float, optional): optimizer learning rates
            device (torch.device, optional): the device to perform computations on
        """
        self.action_dim = action_dim
        self.initial_alpha = initial_alpha
        self.lr = lr
        self.device = device

        self.target = -action_dim
        self.log_alpha = nn.Parameter(torch.tensor(initial_alpha, device=device).log())

        self.optim = optim([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        """
        Get the current entropy coefficient (alpha).

        Returns:
            alpha (torch.Tensor): the entropy coefficient.
        """
        return self.log_alpha.exp()

    def compute_loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the entropy coefficient loss.

        Parameters:
            log_probs (torch.Tensor): log probabilities for actions

        Returns:
            loss (torch.Tensor): the entropy loss value.
        """
        loss = torch.tensor(0.0, device=log_probs.device)

        entropy = (log_probs + self.target).detach()
        loss = -(self.log_alpha * entropy).mean()

        return loss

    def gradient_step(self, loss: torch.Tensor) -> None:
        """
        Performs a gradient update step.

        Parameters:
            loss (torch.Tensor): loss to backpropagate
        """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def config(self) -> EntropyParameters:
        """
        Creates a module config.

        Returns:
            config (EntropyParameters): a parameter config model.
        """
        return EntropyParameters(
            lr=self.lr,
            initial_alpha=self.initial_alpha,
            target=self.target,
        )

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "entropy_optim": self.optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        self.optim.load_state_dict(state_dict["entropy_optim"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"action_dim={self.action_dim}, "
            f"initial_alpha={self.initial_alpha}, "
            f"optim={type(self.optim).__name__}, "
            f"lr={self.lr}, "
            f"device={self.device})"
        )


class EntropyModuleDiscrete(BaseModule):
    """
    An Entropy module for NeuroFlow. Uses automatic entropy tuning from SAC
    based on the paper: [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207).

    Usable with discrete action spaces.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        initial_alpha: float = 1.0,
        optim: Type[optim.Optimizer] = optim.Adam,
        lr: float = 3e-4,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            action_dim (int): dimension of the action space
            initial_alpha (float, optional): the starting entropy coefficient value
            optim (Type[optim.Optimizer], optional): a `PyTorch` optimizer class
            lr (float, optional): optimizer learning rates
            device (torch.device, optional): the device to perform computations on
        """
        self.action_dim = action_dim
        self.initial_alpha = initial_alpha
        self.lr = lr
        self.device = device

        self.target = 0.98 * torch.tensor(1 / action_dim, device=device).log()
        self.log_alpha = nn.Parameter(torch.tensor(initial_alpha, device=device).log())

        self.optim = optim([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        """
        Get the current entropy coefficient (alpha).

        Returns:
            alpha (torch.Tensor): the entropy coefficient.
        """
        return self.log_alpha.exp()

    def compute_loss(
        self, probs: torch.Tensor, log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the entropy coefficient loss.

        Parameters:
            probs (torch.Tensor): probabilities for actions
            log_probs (torch.Tensor): log probabilities for actions

        Returns:
            loss (torch.Tensor): the entropy loss value.
        """
        with torch.no_grad():
            entropy_mean = -torch.sum(probs * log_probs, dim=1).mean()

        loss = self.log_alpha * (entropy_mean - self.target)
        return loss

    def gradient_step(self, loss: torch.Tensor) -> None:
        """
        Performs a gradient update step.

        Parameters:
            loss (torch.Tensor): loss to backpropagate
        """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def config(self) -> EntropyParameters:
        """
        Creates a module config.

        Returns:
            config (EntropyParameters): a parameter config model.
        """
        return EntropyParameters(
            lr=self.lr,
            initial_alpha=self.initial_alpha,
            target=self.target,
        )

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "entropy_optim": self.optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        self.optim.load_state_dict(state_dict["entropy_optim"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"action_dim={self.action_dim}, "
            f"initial_alpha={self.initial_alpha}, "
            f"optim={type(self.optim).__name__}, "
            f"lr={self.lr}, "
            f"device={self.device})"
        )
