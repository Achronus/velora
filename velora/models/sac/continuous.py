from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Self, Tuple, Type

from velora.buffer.experience import BatchExperience
from velora.utils.format import number_to_short

try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # pragma: no cover

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

if TYPE_CHECKING:
    from velora.callbacks import TrainCallback  # pragma: no cover

from velora.buffer.replay import ReplayBuffer
from velora.models.base import NCPModule, RLAgent
from velora.models.config import (
    ModelDetails,
    RLAgentConfig,
    SACExtraParameters,
    TorchConfig,
)
from velora.training.display import training_info
from velora.training.handler import TrainHandler
from velora.utils.restore import load_model, save_model
from velora.utils.torch import soft_update


class SACActor(NCPModule):
    """
    A Liquid NCP Actor Network for the SAC algorithm. Outputs a Gaussian
    distribution over actions.

    Usable with continuous action spaces.
    """

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
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
        x_t = dist.rsample()  #  Reparameterization trick
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
        actions = torch.tanh(mean)

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
        actions = torch.tanh(x_t)  # Bounded: [-1, 1]

        # Calculate log probability, accounting for tanh
        log_prob: torch.Tensor = dist_log_probs - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return actions, log_prob, new_hidden


class SACCritic(NCPModule):
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


class LiquidSAC(RLAgent):
    """
    A Liquid variant of the Soft Actor-Critic (SAC) algorithm for `continuous` action spaces.

    Follows the implementation from the paper:
    [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290).

    And uses enhancements mentioned in the [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) paper.

    !!! note "Decision nodes"

        `inter` and `command` neurons are automatically calculated using:

        ```python
        command_neurons = max(int(0.4 * n_neurons), 1)
        inter_neurons = n_neurons - command_neurons
        ```
    """

    def __init__(
        self,
        state_dim: int,
        n_neurons: int,
        action_dim: int,
        *,
        optim: Type[optim.Optimizer] = optim.Adam,
        buffer_size: int = 1_000_000,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        initial_alpha: float = 1.0,
        log_std: Tuple[float, float] = (-5, 2),
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters:
            state_dim (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes).
            action_dim (int): number of outputs (motor nodes)
            optim (Type[torch.optim.Optimizer], optional): the type of `PyTorch`
                optimizer to use
            buffer_size (int, optional): the maximum size of the `ReplayBuffer`
            actor_lr (float, optional): the actor optimizer learning rate
            critic_lr (float, optional): the critic optimizer learning rate
            alpha_lr (float, optional): the entropy parameter learning rate
            initial_alpha (float, optional): the starting entropy coefficient value
            log_std (Tuple[float, float], optional): `(low, high)` bounds for the
                log standard deviation of the action distribution. Controls the
                variance of actions
            device (torch.device, optional): the device to perform computations on
            seed (int, optional): random number seed for experiment
                reproducibility. When `None` generates a seed automatically
        """
        super().__init__(state_dim, n_neurons, action_dim, buffer_size, device, seed)

        self.initial_alpha = initial_alpha
        self.log_std = log_std

        self.actor = SACActor(
            self.state_dim,
            self.n_neurons,
            self.action_dim,
            log_std_min=log_std[0],
            log_std_max=log_std[1],
            device=self.device,
        ).to(self.device)

        self.critic = SACCritic(
            self.state_dim,
            self.n_neurons,
            self.action_dim,
            device=self.device,
        ).to(self.device)

        self.critic2 = SACCritic(
            self.state_dim,
            self.n_neurons,
            self.action_dim,
            device=self.device,
        ).to(self.device)

        self.hidden_dim = self.actor.ncp.hidden_size

        self.critic_target = deepcopy(self.critic)
        self.critic2_target = deepcopy(self.critic2)

        self.actor_optim = optim(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim(self.critic.parameters(), lr=critic_lr)
        self.critic2_optim = optim(self.critic2.parameters(), lr=critic_lr)

        # Entropy tuning parameters
        self.target_entropy = -action_dim
        self.log_alpha = nn.Parameter(torch.tensor(initial_alpha, device=device).log())
        self.alpha_optim = optim([self.log_alpha], lr=alpha_lr)

        self.loss = nn.MSELoss()
        self.buffer: ReplayBuffer = ReplayBuffer(
            buffer_size,
            state_dim,
            action_dim,
            self.hidden_dim,
            device=self.device,
        )

        self.active_params = (
            self.actor.ncp.active_params
            + self.critic.ncp.active_params
            + self.critic2.ncp.active_params
        )
        self.total_params = (
            self.actor.ncp.total_params
            + self.critic.ncp.total_params
            + self.critic2.ncp.total_params
        )

        # Init config details
        self.config = RLAgentConfig(
            agent=self.__class__.__name__,
            seed=self.seed,
            model_details=ModelDetails(
                type="actor-critic",
                **locals(),
                target_networks=True,
                exploration_type="Entropy",
                actor=self.actor.config(),
                critic=self.critic.config(),
                critic2=self.critic2.config(),
                extras=SACExtraParameters(
                    alpha_lr=alpha_lr,
                    initial_alpha=initial_alpha,
                    target_entropy=self.target_entropy,
                    log_std_min=log_std[0],
                    log_std_max=log_std[1],
                ),
            ),
            buffer=self.buffer.config(),
            torch=TorchConfig(
                device=str(self.device),
                optimizer=optim.__name__,
                loss=self.loss.__class__.__name__,
            ),
        )

        self.actor: SACActor = torch.jit.script(self.actor)
        self.critic: SACCritic = torch.jit.script(self.critic)
        self.critic2: SACCritic = torch.jit.script(self.critic2)

        self.critic_target: SACCritic = torch.jit.script(self.critic_target)
        self.critic2_target: SACCritic = torch.jit.script(self.critic2_target)

    @property
    def alpha(self) -> torch.Tensor:
        """
        Get the current entropy coefficient (alpha).

        Returns:
            alpha (torch.Tensor): the entropy coefficient.
        """
        return self.log_alpha.exp()

    def _update_target_networks(self, tau: float) -> None:
        """
        Helper method. Performs a soft update on the target networks.

        Parameters:
            tau (float): a soft decay coefficient for updating the target network
                weights
        """
        soft_update(self.critic, self.critic_target, tau=tau)
        soft_update(self.critic2, self.critic2_target, tau=tau)

    def _update_critics(self, batch: BatchExperience, gamma: float) -> torch.Tensor:
        """
        Helper method. Performs Critic network updates.

        Parameters:
            batch (BatchExperience): an object containing a batch of experience
                with `(states, actions, rewards, next_states, dones, hidden)`
                from the buffer
            gamma (float): the reward discount factor

        Returns:
            critic_loss (torch.Tensor): total Critic network loss `(c1_loss + c2_loss)`.
        """
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor(
                batch.next_states, batch.hiddens
            )

            # Q-values predictions from critics
            next_q1, _ = self.critic_target(batch.next_states, next_actions)
            next_q2, _ = self.critic2_target(batch.next_states, next_actions)

            # Compute target Q-value
            next_q = torch.min(next_q1, next_q2)  # Mitigate overestimation bias
            next_q = next_q - self.alpha * next_log_probs
            target_q = batch.rewards + (1 - batch.dones) * gamma * next_q

        current_q1, _ = self.critic(batch.states, batch.actions)
        current_q2, _ = self.critic2(batch.states, batch.actions)

        # Calculate loss
        c1_loss: torch.Tensor = self.loss(current_q1, target_q)
        c2_loss: torch.Tensor = self.loss(current_q2, target_q)
        critic_loss: torch.Tensor = c1_loss + c2_loss

        # Update critics
        self.critic_optim.zero_grad()
        c1_loss.backward()
        self.critic_optim.step()

        self.critic2_optim.zero_grad()
        c2_loss.backward()
        self.critic2_optim.step()

        return critic_loss

    @staticmethod
    @torch.jit.script
    def _actor_loss(
        q1: torch.Tensor,
        q2: torch.Tensor,
        alpha: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper method. Computes the Actor loss.

        Parameters:
            q1 (torch.Tensor): first Critic networks Q-Value predictions
            q2 (torch.Tensor): second Critic networks Q-Value predictions
            alpha (torch.Tensor): current entropy coefficient
            log_probs (torch.Tensor): Actor log probabilities for actions

        Returns:
            actor_loss (torch.Tensor): the Actor's loss value.
        """
        min_q = torch.min(q1, q2)

        # Compute actor loss (-Q-value + entropy)
        actor_loss: torch.Tensor = (alpha * log_probs - min_q).mean()

        return actor_loss

    @staticmethod
    @torch.jit.script
    def _entropy_loss(
        log_probs: torch.Tensor,
        target_entropy: int,
        log_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper method. Computes the entropy coefficient loss.

        Parameters:
            log_probs (torch.Tensor): Actor log probabilities for actions
            target_entropy (int): level of entropy in the policy
            log_alpha (torch.Tensor): current log entropy coefficient

        Returns:
            entropy_loss (torch.Tensor): the entropy loss value.
        """
        loss = torch.tensor(0.0, device=log_probs.device)

        entropy = (log_probs + target_entropy).detach()
        loss = -(log_alpha * entropy).mean()

        return loss

    def _train_step(
        self,
        batch_size: int,
        gamma: float,
        tau: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Helper method. Performs a single training step.

        Parameters:
            batch_size (int): number of samples in a batch
            gamma (float): the reward discount factor
            tau (float): soft target network update factor

        Returns:
            losses (Dict[str, torch.Tensor]): a dictionary of losses -

            - critic: the total critic loss.
            - actor: the actor loss.
            - entropy: the entropy loss.
        """
        if len(self.buffer) < batch_size:
            return {
                "critic": torch.zeros(1),
                "actor": torch.zeros(1),
                "entropy": torch.zeros(1),
            }

        batch = self.buffer.sample(batch_size)

        # Compute critic loss
        critic_loss = self._update_critics(batch, gamma)

        # Make predictions
        actions, log_probs, _ = self.actor(batch.states, batch.hiddens)
        q1, _ = self.critic(batch.states, actions)
        q2, _ = self.critic2(batch.states, actions)

        # Compute actor and entropy losses
        actor_loss: torch.Tensor = self._actor_loss(q1, q2, self.alpha, log_probs)
        entropy_loss: torch.Tensor = self._entropy_loss(
            log_probs,
            self.target_entropy,
            self.log_alpha,
        )

        # Update gradients
        self._gradient_step(actor_loss, entropy_loss)

        # Update target networks
        self._update_target_networks(tau)

        return {
            "critic": critic_loss.detach(),
            "actor": actor_loss.detach(),
            "entropy": entropy_loss.detach(),
        }

    def _gradient_step(
        self,
        actor_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
    ) -> None:
        """
        Performs a gradient update step for non-critic networks.

        Parameters:
            actor_loss (torch.Tensor): Actor network loss
            entropy_loss (torch.Tensor): entropy loss
        """
        # Update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update alpha
        self.alpha_optim.zero_grad()
        entropy_loss.backward()
        self.alpha_optim.step()

    @override
    def train(
        self,
        env: gym.Env,
        batch_size: int,
        *,
        n_episodes: int = 1000,
        callbacks: List["TrainCallback"] | None = None,
        display_count: int = 100,
        window_size: int = 100,
        max_steps: int = 1000,
        gamma: float = 0.99,
        tau: float = 0.005,
        warmup_steps: int = 1000,
    ) -> None:
        """
        Trains the agent on a Gymnasium environment using a `ReplayBuffer`.

        Parameters:
            env (gym.Env): the Gymnasium environment to train on
            batch_size (int): the number of features in a single batch
            n_episodes (int, optional): the total number of episodes to train for
            callbacks (List[TrainCallback], optional): a list of training callbacks
                that are applied during the training process
            display_count (int, optional): console training progress frequency
                (in episodes)
            window_size (int, optional): the reward moving average size
                (in episodes)
            max_steps (int, optional): the total number of steps per episode
            gamma (float, optional): the reward discount factor
            tau (float, optional): the soft update factor used to slowly update
                the target networks
            warmup_steps (int, optional): number of random steps to take before
                starting training
        """
        if not isinstance(env.action_space, gym.spaces.Box):
            raise EnvironmentError(
                f"Invalid '{env.action_space=}'. Must be 'gym.spaces.Box'."
            )

        # Add training details to config
        self.config = self.config.update(
            env.spec.name,
            self._set_train_params(locals()),
        )

        # Display console details
        env.reset(seed=self.seed)  # Set seed
        training_info(
            self,
            env.spec.id,
            n_episodes,
            batch_size,
            window_size,
            callbacks or [],
            self.device,
            env.np_random_seed,
            extras=f"Warming buffer with '{number_to_short(warmup_steps)}' samples before training starts.\n",
        )

        self.buffer.warm(self, env.spec.id, warmup_steps, self.seed)

        with TrainHandler(
            self, env, n_episodes, max_steps, window_size, callbacks
        ) as handler:
            for current_ep in range(1, n_episodes + 1):
                ep_reward = 0.0
                hidden = None

                state, _ = handler.env.reset()

                for current_step in range(1, max_steps + 1):
                    action, hidden = self.predict(state, hidden, train_mode=True)
                    next_state, reward, terminated, truncated, info = handler.env.step(
                        action
                    )
                    done = terminated or truncated

                    self.buffer.add(state, action, reward, next_state, done, hidden)

                    losses = self._train_step(batch_size, gamma, tau)

                    handler.metrics.add_step(**losses)
                    handler.step(current_step)

                    state = next_state

                    if done:
                        ep_reward = info["episode"]["r"].item()
                        handler.metrics.add_episode(
                            current_ep,
                            info["episode"]["r"],
                            info["episode"]["l"],
                        )
                        break

                if (
                    current_ep % display_count == 0
                    or current_ep == n_episodes
                    or handler.stop()
                ):
                    handler.metrics.info(current_ep)

                handler.episode(current_ep, ep_reward)

                # Terminate on early stopping
                if handler.stop():
                    break

    @override
    def predict(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        train_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Makes an action prediction using the Actor network.

        Parameters:
            state (torch.Tensor): the current state
            hidden (torch.Tensor, optional): the current hidden state
            train_mode (bool, optional): whether to make deterministic (when
                `False`) or stochastic (when `True`) action predictions

        Returns:
            action (torch.Tensor): the action prediction on the given state
            hidden (torch.Tensor): the Actor network's new hidden state
        """
        self.actor.eval()
        with torch.no_grad():
            state = state.unsqueeze(0) if state.dim() < 2 else state

            if not train_mode:
                action, hidden = self.actor.predict(state, hidden)
            else:
                action, _, hidden = self.actor(state, hidden)

        self.actor.train()
        return action, hidden

    def save(
        self,
        dirpath: str | Path,
        *,
        buffer: bool = False,
        config: bool = False,
    ) -> None:
        save_model(self, dirpath, buffer=buffer, config=config)

    @classmethod
    def load(cls, dirpath: str | Path, *, buffer: bool = False) -> Self:
        return load_model(cls, dirpath, buffer=buffer)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"n_neurons={self.n_neurons}, "
            f"action_dim={self.action_dim}, "
            f"optim={type(self.actor_optim).__name__}, "
            f"buffer_size={self.buffer_size:,}, "
            f"initial_alpha={self.initial_alpha}, "
            f"log_std={self.log_std}, "
            f"device={self.device})"
        )
