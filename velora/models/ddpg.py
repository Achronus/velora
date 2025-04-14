from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Self, Tuple, Type

try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # pragma: no cover

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from velora.callbacks import TrainCallback  # pragma: no cover
    from velora.buffer.experience import BatchExperience  # pragma: no cover

from velora.buffer.replay import ReplayBuffer
from velora.models.base import LiquidNCPModule, RLAgent
from velora.models.config import BasicModelDetails, RLAgentConfig, TorchConfig
from velora.noise import OUNoise
from velora.training.display import training_info
from velora.training.handler import TrainHandler
from velora.utils.restore import load_model, save_model
from velora.utils.torch import soft_update

CheckpointLiteral = Literal[
    "state_dim",
    "n_neurons",
    "action_dim",
    "buffer_size",
    "device",
    "actor",
    "critic",
    "actor_target",
    "critic_target",
    "actor_optim",
    "critic_optim",
]


class DDPGActor(LiquidNCPModule):
    """
    A Liquid NCP Actor Network for the DDPG algorithm.
    """

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        *,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            num_actions (int): the number of actions
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(num_obs, n_neurons, num_actions, device=device)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the hidden state

        Returns:
            actions (torch.Tensor): the action predictions.
            hidden (torch.Tensor): the new hidden state.
        """
        actions, new_hidden = self.ncp(obs, hidden)
        scaled_actions = torch.tanh(actions)  # Bounded: [-1, 1]
        return scaled_actions, new_hidden


class DDPGCritic(LiquidNCPModule):
    """
    A Liquid NCP Critic Network for the DDPG algorithm.
    """

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        *,
        device: torch.device | None = None,
    ):
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


class LiquidDDPG(RLAgent):
    """
    A Liquid variant of the Deep Deterministic Policy Gradient (DDPG)
    algorithm from the paper: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971).

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
        buffer_size: int = 100_000,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
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
            device (torch.device, optional): the device to perform computations on
            seed (int, optional): random number seed for experiment reproducibility.
                When `None` generates a seed automatically
        """
        super().__init__(state_dim, n_neurons, action_dim, buffer_size, device, seed)

        self.actor = DDPGActor(
            self.state_dim,
            self.n_neurons,
            self.action_dim,
            device=self.device,
        ).to(self.device)

        self.critic = DDPGCritic(
            self.state_dim,
            self.n_neurons,
            self.action_dim,
            device=self.device,
        ).to(self.device)

        self.hidden_dim = self.actor.ncp.hidden_size

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        critic_extras = (
            dict(weight_decay=1e-2) if isinstance(optim, torch.optim.Adam) else {}
        )

        self.actor_optim = optim(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim(
            self.critic.parameters(), lr=critic_lr, **critic_extras
        )

        self.loss = nn.MSELoss()
        self.buffer: ReplayBuffer = ReplayBuffer(
            buffer_size,
            state_dim,
            action_dim,
            self.hidden_dim,
            device=self.device,
        )
        self.noise = OUNoise(action_dim, device=device)

        self.active_params = (
            self.actor.ncp.active_params + self.critic.ncp.active_params
        )
        self.total_params = self.actor.ncp.total_params + self.critic.ncp.total_params

        # Init config details
        self.config = RLAgentConfig(
            agent=self.__class__.__name__,
            seed=self.seed,
            model_details=BasicModelDetails(
                type="actor-critic",
                **locals(),
                target_networks=True,
                exploration_type="OUNoise",
                actor=self.actor.config(),
                critic=self.critic.config(),
            ),
            buffer=self.buffer.config(),
            torch=TorchConfig(
                device=str(self.device),
                optimizer=optim.__name__,
                loss=self.loss.__class__.__name__,
            ),
        )

        self.actor: DDPGActor = torch.jit.script(self.actor)
        self.critic: DDPGCritic = torch.jit.script(self.critic)

    def _update_target_networks(self, tau: float) -> None:
        """
        Helper method. Performs a soft update on the target networks.

        Parameters:
            tau (float): a soft decay coefficient for updating the target network
                weights
        """
        soft_update(self.actor, self.actor_target, tau=tau)
        soft_update(self.critic, self.critic_target, tau=tau)

    def _update_critic(self, batch: "BatchExperience", gamma: float) -> torch.Tensor:
        """
        Helper method. Performs a Critic Network update.

        Parameters:
            batch (BatchExperience): an object containing a batch of experience
                with `(states, actions, rewards, next_states, dones, hidden)`
                from the buffer
            gamma (float): the reward discount factor

        Returns:
            critic_loss (torch.Tensor): the Critic's loss value.
        """
        with torch.no_grad():
            next_states = batch.next_states
            next_actions, _ = self.actor_target(next_states, batch.hiddens)
            target_q, _ = self.critic_target(next_states, next_actions)
            target_q = batch.rewards + (1 - batch.dones) * gamma * target_q

        current_q, _ = self.critic(batch.states, batch.actions)
        critic_loss: torch.Tensor = self.loss(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss

    def _update_actor(self, batch: "BatchExperience") -> torch.Tensor:
        """
        Helper method. Performs an Actor Network update.

        Parameters:
            batch (BatchExperience): an object containing a batch of experience
                with `(states, actions, rewards, next_states, dones, hidden,
                critic_hidden)` from the buffer

        Returns:
            actor_loss (torch.Tensor): the Actor's loss value.
        """
        next_actions, _ = self.actor(batch.states, batch.hiddens)
        actor_q, _ = self.critic(batch.states, next_actions)
        actor_loss: torch.Tensor = -actor_q.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return actor_loss

    def _train_step(
        self, batch_size: int, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method. Performs a single training step.

        Parameters:
            batch_size (int): number of samples in a batch
            gamma (float): the reward discount factor

        Returns:
            critic_loss (torch.Tensor): the critic loss.
            actor_loss (torch.Tensor): the actor loss.
        """
        if len(self.buffer) < batch_size:
            return

        batch = self.buffer.sample(batch_size)

        critic_loss = self._update_critic(batch, gamma)
        actor_loss = self._update_actor(batch)

        return critic_loss, actor_loss

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
        noise_scale: float = 0.3,
        gamma: float = 0.99,
        tau: float = 0.005,
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
            noise_scale (float, optional): the exploration noise added when
                selecting an action
            gamma (float, optional): the reward discount factor
            tau (float, optional): the soft update factor used to slowly update
                the target networks
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
        )

        self.noise.reset()
        self.buffer.warm(self, env.spec.id, batch_size, self.seed)

        with TrainHandler(
            self, env, n_episodes, max_steps, window_size, callbacks
        ) as handler:
            for current_ep in range(1, n_episodes + 1):
                ep_reward = 0.0
                hidden = None

                state, _ = handler.env.reset()
                self.noise.reset()

                for current_step in range(1, max_steps + 1):
                    action, hidden = self.predict(
                        state,
                        hidden,
                        noise_scale=noise_scale,
                        train_mode=True,
                    )
                    next_state, reward, terminated, truncated, info = handler.env.step(
                        action
                    )
                    done = terminated or truncated

                    self.buffer.add(state, action, reward, next_state, done, hidden)

                    critic_loss, actor_loss = self._train_step(batch_size, gamma)
                    self._update_target_networks(tau)

                    handler.metrics.add_step(critic_loss, actor_loss)
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
        noise_scale: float = 0.3,
        train_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Makes an action prediction using the Actor network with exploration noise.

        Parameters:
            state (torch.Tensor): the current state
            hidden (torch.Tensor, optional): the current hidden state
            noise_scale (float, optional): the exploration noise added when
                selecting an action
            train_mode (bool, optional): whether the model is in training model

        Returns:
            action (torch.Tensor): the action prediction on the given state
            hidden (torch.Tensor): the Actor networks new hidden state
        """
        self.actor.eval()
        with torch.no_grad():
            state = state.unsqueeze(0) if state.dim() < 2 else state
            action, hidden = self.actor(state, hidden)

            if train_mode and noise_scale > 0:
                # Exploration noise
                noise = self.noise.sample() * noise_scale
                action = torch.clamp(action + noise, min=-1, max=1)

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
        return f"{self.__class__.__name__}(state_dim={self.state_dim}, n_neurons={self.n_neurons}, action_dim={self.action_dim}, optim={type(self.actor_optim).__name__}, buffer_size={self.buffer_size:,}, device={self.device})"
