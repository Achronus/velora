from copy import deepcopy
from typing import Tuple, Type

from velora.buffer import BatchExperience, Experience, ReplayBuffer
from velora.models import LiquidNCPNetwork
from velora.models.utils import soft_update

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import numpy as np

from velora.noise import OUNoise


class DDPGActor(nn.Module):
    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        *,
        device: torch.device | None = None,
    ):
        """
        An Actor Network for the DDPG algorithm.

        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            num_actions (int): the number of actions
            device (torch.device, optional): the device to load `torch.Tensors`
                onto. Default is `None`
        """

        super().__init__()

        self.ncp = LiquidNCPNetwork(
            in_features=num_obs,
            n_neurons=n_neurons,
            out_features=num_actions,
            device=device,
        )

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the hidden state. Default is `None`

        Returns:
            actions,hidden (Tuple[torch.Tensor, torch.Tensor]): returns the action
            predictions and new hidden state.
        """
        actions, new_hidden = self.ncp.forward(obs, hidden)
        scaled_actions = torch.tanh(actions)  # Bounded: [-1, 1]
        return scaled_actions, new_hidden


class DDPGCritic(nn.Module):
    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        *,
        device: torch.device | None = None,
    ):
        """
        A Critic Network for the DDPG algorithm.

        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            num_actions (int): the number of actions
            device (torch.device, optional): the device to load `torch.Tensors`
                onto. Default is `None`
        """
        super().__init__()

        self.ncp = LiquidNCPNetwork(
            in_features=num_obs + num_actions,
            n_neurons=n_neurons,
            out_features=1,  # Q-value output
            device=device,
        )

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
            hidden (torch.Tensor, optional): the hidden state. Default is `None`

        Returns:
            q_values,hidden (Tuple[torch.Tensor, torch.Tensor]): returns the Q-Value
            predictions and new hidden state.
        """
        inputs = torch.cat([obs, actions], dim=-1)

        q_values, new_hidden = self.ncp.forward(inputs, hidden)
        return q_values, new_hidden


class LiquidDDPG:
    """
    A Liquid Network variant of the Deep Deterministic Policy Gradient (DDPG)
    algorithm from: https://arxiv.org/abs/1509.02971.

    Parameters:
        state_dim (int): number of inputs (sensory nodes)
        n_neurons (int): number of decision nodes (inter and command nodes).
            Nodes are set automatically based on the following:
            ```python
            command_neurons = max(int(0.4 * n_neurons), 1)
            inter_neurons = n_neurons - command_neurons
            ```
        action_dim (int): number of outputs (motor nodes)
        optim (Type[torch.optim.Optimizer], optional): the type of `PyTorch`
            optimizer to use. Default is `torch.optim.Adam`
        buffer_size (int, optional): the maximum size of the ReplayBuffer.
            Default is `100_000`
        actor_lr (float, optional): the actor optimizer learning rate.
            Default is `1e-4`
        critic_lr (float, optional): the critic optimizer learning rate.
            Default is `1e-3`
        device (torch.device, optional): the device to load `torch.Tensors` onto.
            Default is `None`
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
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.n_neurons = n_neurons
        self.action_dim = action_dim
        self.device = device

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

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        # Freeze target networks
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim = optim(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim(self.critic.parameters(), lr=critic_lr)

        self.loss = nn.MSELoss()
        self.buffer = ReplayBuffer(capacity=buffer_size, device=device)
        self.noise = OUNoise(action_dim, device=device)

    def _update_target_networks(self, tau: float) -> None:
        """Performs a soft update on the target networks."""
        soft_update(self.actor, self.actor_target, tau)
        soft_update(self.critic, self.critic_target, tau)

    def _update_critic(self, batch: BatchExperience, gamma: float) -> float:
        """
        Performs a Critic Network update.

        Returns:
            critic_loss (float): the Critic's loss value.
        """
        with torch.no_grad():
            next_states = batch.next_states
            next_actions, _ = self.actor_target.forward(next_states)
            target_q, _ = self.critic_target.forward(next_states, next_actions)
            target_q = batch.rewards + (1 - batch.dones) * gamma * target_q

        current_q, _ = self.critic.forward(batch.states, batch.actions.unsqueeze(1))
        critic_loss = self.loss.forward(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss.item()

    def _update_actor(self, states: torch.Tensor) -> float:
        """
        Performs an Actor Network update.

        Returns:
            actor_loss (float): the Actor's loss value.
        """
        next_actions, _ = self.actor.forward(states)
        actor_q, _ = self.critic.forward(states, next_actions)
        actor_loss = -actor_q.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return actor_loss.item()

    def _train_step(self, batch_size: int, gamma: float) -> Tuple[float, float]:
        """Performs a single training step."""
        if len(self.buffer) < batch_size:
            return

        batch: BatchExperience = self.buffer.sample(batch_size)

        critic_loss = self._update_critic(batch, gamma)
        actor_loss = self._update_actor(batch.states)

        return critic_loss, actor_loss

    def train(
        self,
        env: gym.Env,
        batch_size: int,
        *,
        n_episodes: int = 1000,
        max_steps: int = 1000,
        noise_scale: float = 0.1,
        gamma: float = 0.99,
        tau: float = 0.005,
        output_count: int = 100,
    ) -> None:
        """
        Trains the agent.

        Parameters:
            env (gym.Env): the Gymnasium environment to train on
            batch_size (int): the number of features in a single batch
            n_episodes (int, optional): the total number of episodes to train for.
                Default is `1000`
            max_steps (int, optional): the total number of steps per episode.
                Default is `1000`
            noise_scale (float, optional): the exploration noise added when
                selecting an action. Default is `0.1`
            gamma (float, optional): the reward discount factor. Default is `0.99`
            tau (float, optional): the soft update factor used to slowly update
                the target networks. Default is `0.005`
            output_count (int, optional): the episodic rate for displaying
                information to the console. Default is `100`
        """
        if not isinstance(env.action_space, gym.spaces.Box):
            raise EnvironmentError(
                f"Invalid '{env.action_space=}'. Must be 'gym.spaces.Box'."
            )

        episode_rewards = []
        training_started = False

        print(f"{batch_size=}, getting buffer samples.")
        for i_ep in range(n_episodes):
            state, _ = env.reset()

            episode_reward = 0
            critic_losses, actor_losses = [], []
            actor_hidden = None

            for i_step in range(max_steps):
                action, actor_hidden = self.predict(
                    state,
                    actor_hidden,
                    noise_scale=noise_scale,
                )
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.push(
                    Experience(state, action.item(), reward, next_state, done),
                )

                if len(self.buffer) >= batch_size:
                    if not training_started:
                        print("Buffer warmed. Starting training...")
                        training_started = True

                    critic_loss, actor_loss = self._train_step(batch_size, gamma)
                    self._update_target_networks(tau)

                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)

                state = next_state
                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)

            if training_started and (i_ep + 1) % output_count == 0:
                avg_reward = np.mean(episode_rewards[-output_count:])
                avg_critic_loss = np.mean(critic_losses)
                avg_actor_loss = np.mean(actor_losses)

                print(
                    f"Episode: {i_ep + 1}/{n_episodes}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Critic Loss: {avg_critic_loss:.2f}, "
                    f"Actor Loss: {avg_actor_loss:.2f}"
                )

        return episode_rewards

    def predict(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor = None,
        *,
        noise_scale: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Makes an action prediction using the Actor network with exploration noise.

        Parameters:
            state (torch.Tensor): the current state
            hidden (torch.Tensor, optional): the current hidden state.
                Default is `None`
            noise_scale (float, optional): the exploration noise added when
                selecting an action. Default is `0.1`

        Returns:
            action,hidden (Tuple[torch.Tensor, torch.Tensor]): the action prediction
            and hidden state.
        """
        self.actor.eval()
        with torch.no_grad():
            action, hidden = self.actor.forward(state.unsqueeze(0), hidden)

            if noise_scale > 0:
                # Exploration noise
                noise = self.noise.sample() * noise_scale
                action = torch.clamp(action + noise, min=-1, max=1)

        self.actor.train()
        return action.flatten(), hidden
