from typing import TYPE_CHECKING, List, Tuple, Type

from velora.buffer.experience import RolloutBatchExperience
from velora.buffer.rollout import RolloutBuffer
from velora.models.base import NCPModule, RLAgent
from velora.models.config import ModelDetails, RLAgentConfig, TorchConfig
from velora.utils.compute import closest_divisible, get_mini_batch_size

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

from velora.training.display import vec_training_info
from velora.training.handler import VecTrainHandler


class PPOActor(NCPModule):
    """
    A Liquid NCP Actor Network for the PPO algorithm.
    """

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        num_actions: int,
        *,
        discrete: bool = False,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            num_actions (int): the number of actions
            discrete (bool, optional): whether the action space is discrete
            device (torch.device, optional): the device to perform computations on
        """
        output_size = num_actions if discrete else num_actions * 2
        super().__init__(num_obs, n_neurons, output_size, device=device)

        self.num_actions = num_actions
        self.discrete = discrete

    @torch.jit.ignore
    def _discrete(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper method. Performs discrete action predictions.

        Parameters:
            logits (torch.Tensor): the network logits (unnormalized
                network predictions)

        Returns:
            y_pred (torch.Tensor): action predictions.
            log_probs (torch.Tensor): the log probabilities of the actions.
            entropy (torch.Tensor): the entropy of the action distribution.
        """
        dist = torch.distributions.Categorical(logits=logits)

        if self.training:
            actions = dist.sample()
        else:
            actions = torch.argmax(logits, dim=-1)

        log_probs: torch.Tensor = dist.log_prob(actions).unsqueeze(-1)
        entropy: torch.Tensor = dist.entropy().unsqueeze(-1)

        return actions.unsqueeze(-1), log_probs, entropy

    @torch.jit.ignore
    def _continuous(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper method. Performs continuous action predictions.

        Parameters:
            x (torch.Tensor): the network output

        Returns:
            y_pred (torch.Tensor): action predictions.
            log_probs (torch.Tensor): the log probabilities of the actions.
            entropy (torch.Tensor): the entropy of the action distribution.
        """
        means, log_stds = torch.split(x, self.num_actions, dim=-1)
        log_stds = torch.clamp(log_stds, -20, 2)

        std = torch.exp(log_stds)
        dist = torch.distributions.Normal(means, std)

        if self.training:
            actions = dist.sample()
        else:
            actions = means

        # Clip actions to [-1, 1]
        actions = torch.tanh(actions)

        log_probs: torch.Tensor = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy: torch.Tensor = dist.entropy().sum(-1, keepdim=True)

        return actions, log_probs, entropy

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            obs (torch.Tensor): the batch of state observations
            hidden (torch.Tensor, optional): the hidden state

        Returns:
            y_pred (torch.Tensor): action predictions.
            log_probs (torch.Tensor): the log probabilities of the actions.
            entropy (torch.Tensor): the entropy of the action distribution.
            hidden (torch.Tensor): the new hidden state.
        """
        x, new_hidden = self.ncp(obs, hidden)

        if self.discrete:
            y_pred, log_probs, entropy = self._discrete(x)
        else:
            y_pred, log_probs, entropy = self._continuous(x)

        return y_pred, log_probs, entropy, new_hidden


class PPOCritic(NCPModule):
    """
    A Liquid NCP Critic Network for the PPO algorithm.
    """

    def __init__(
        self,
        num_obs: int,
        n_neurons: int,
        *,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            num_obs (int): the number of input observations
            n_neurons (int): the number of hidden neurons
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(num_obs, n_neurons, 1, device=device)

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
            values (torch.Tensor): the value function estimates.
            hidden (torch.Tensor): the new hidden state.
        """
        values, new_hidden = self.ncp(obs, hidden)
        return values, new_hidden


class LiquidPPO(RLAgent):
    """
    A Liquid variant of the Proximal Policy Optimization (PPO)
    algorithm from the paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

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
        discrete: bool = False,
        optim: Type[optim.Optimizer] = optim.Adam,
        buffer_size: int = 2048,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            state_dim (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes).
            action_dim (int): number of outputs (motor nodes)
            discrete (bool, optional): whether the action space is discrete
            optim (Type[torch.optim.Optimizer], optional): the type of `PyTorch`
                optimizer to use
            buffer_size (int, optional): the maximum size of the `RolloutBuffer`
            actor_lr (float, optional): the actor optimizer learning rate
            critic_lr (float, optional): the critic optimizer learning rate
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(state_dim, n_neurons, action_dim, buffer_size, device)

        self.discrete = discrete
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = PPOActor(
            self.state_dim,
            self.n_neurons,
            self.action_dim,
            discrete=discrete,
            device=self.device,
        ).to(self.device)

        self.critic = PPOCritic(
            self.state_dim,
            self.n_neurons,
            device=self.device,
        ).to(self.device)

        self.actor_optim = optim(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim(self.critic.parameters(), lr=critic_lr)

        self.loss = nn.MSELoss()
        self.buffer: RolloutBuffer = RolloutBuffer(
            buffer_size,
            state_dim,
            action_dim,
            device=self.device,
        )

        self.active_params = (
            self.actor.ncp.active_params + self.critic.ncp.active_params
        )
        self.total_params = self.actor.ncp.total_params + self.critic.ncp.total_params

        # Init config details
        self.config = RLAgentConfig(
            agent=self.__class__.__name__,
            model_details=ModelDetails(
                type="actor-critic",
                **locals(),
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

        self.actor: PPOActor = torch.jit.script(self.actor)
        self.critic: PPOCritic = torch.jit.script(self.critic)

    def _anneal_lr(self, i_update: int, total_updates: int) -> None:
        """
        Helper method. Performs learning rate annealing on the network optimizers.

        Parameters:
            i_update (int): current update index
            total_updates (int): total number of updates
        """
        frac = 1.0 - (i_update - 1.0) / total_updates
        critic_new_lr = frac * self.critic_lr
        actor_new_lr = frac * self.actor_lr

        self.actor_optim.param_groups[0]["lr"] = actor_new_lr
        self.critic_optim.param_groups[0]["lr"] = critic_new_lr

    def _compute_advantages(
        self,
        batch: RolloutBatchExperience,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method. Computes advantages using Generalized Advantage
        Estimation (GAE).

        Parameters:
            batch (RolloutBatchExperience): an object containing a batch of
                experience from the buffer
            gamma (float): the reward discount factor
            gae_lambda (float): the GAE smoothing parameter

        Returns:
            advantages (torch.Tensor): the calculated advantages.
            returns (torch.Tensor): the calculated returns (for value function update).
        """
        with torch.no_grad():
            next_values, _ = self.critic(batch.next_states)

            rewards = batch.rewards
            values = batch.values
            dones = batch.dones

            # Create mask for non-terminals
            non_terminals = 1.0 - dones

            # Compute delta terms: δt = rt + γVt+1 - Vt (TD error)
            deltas = rewards + gamma * next_values * non_terminals - values
            advantages = torch.zeros_like(rewards, device=self.device)
            discount = gamma * gae_lambda

            last_gae = torch.zeros(1, device=self.device)
            batch_size = rewards.size(0)

            # TODO: try compile this
            # Process in reverse order (GAE)
            for t in range(batch_size - 1, -1, -1):
                last_gae = deltas[t] + discount * non_terminals[t] * last_gae
                advantages[t] = last_gae

            returns = advantages + values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return advantages, returns

    def _update_actor(
        self,
        batch: RolloutBatchExperience,
        advantages: torch.Tensor,
        clip_ratio: float,
        entropy_coef: float,
        grad_clip: float,
    ) -> torch.Tensor:
        """
        Helper method. Performs an Actor network update.

        Parameters:
            batch (RolloutBatchExperience): an object containing a batch of
                experience from the buffer
            advantages (torch.Tensor): advantage values
            clip_ratio (float): the surrogate clipping ratio
            entropy_coef (float): entropy exploration coefficient
            grad_clip (float): max norm gradient clip

        Returns:
            loss (torch.Tensor): the Actor's loss value.
        """
        _, new_log_probs, entropy, _ = self.actor(batch.states)

        # Compute policy ratio
        ratio = torch.exp(new_log_probs - batch.log_probs)

        # Clip surrogate objective
        clip = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clip * advantages).mean()

        # Entropy bonus (encourages exploration)
        entropy_loss = -entropy.mean() * entropy_coef

        # Total loss
        actor_loss: torch.Tensor = policy_loss + entropy_loss

        # Compute gradients and update network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=grad_clip)
        self.actor_optim.step()

        return actor_loss

    def _update_critic(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
        grad_clip: float,
    ) -> torch.Tensor:
        """
        Helper method. Performs a Critic network update.

        Parameters:
            states (torch.Tensor): current mini-batch states
            returns (torch.Tensor): the calculated returns (advantages + values)
            grad_clip (float): max norm gradient clip

        Returns:
            loss (torch.Tensor): the Critic's loss value.
        """
        values, _ = self.critic(states)
        critic_loss: torch.Tensor = self.loss(values, returns)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=grad_clip)
        self.critic_optim.step()

        return critic_loss

    def _train_step(
        self,
        batch: RolloutBatchExperience,
        gamma: float,
        gae_lambda: float,
        clip_ratio: float,
        entropy_coef: float,
        grad_clip: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method. Performs a single policy update (training) step.

        Parameters:
            batch (RolloutBatchExperience): an object containing a batch of
                experience from the buffer
            gamma (float): the reward discount factor
            gae_lambda (float): the GAE smoothing parameter
            clip_ratio (float): the surrogate clipping ratio
            entropy_coef (float): entropy exploration coefficient
            grad_clip (float): max norm gradient clip

        Returns:
            critic_loss (torch.Tensor): the critic loss.
            actor_loss (torch.Tensor): the actor loss.
        """
        advantages, returns = self._compute_advantages(batch, gamma, gae_lambda)

        actor_loss = self._update_actor(
            batch,
            advantages,
            clip_ratio,
            entropy_coef,
            grad_clip,
        )
        critic_loss = self._update_critic(batch.states, returns, grad_clip)

        return critic_loss, actor_loss

    @override
    def train(
        self,
        envs: gym.vector.VectorEnv,
        batch_size: int,
        *,
        n_steps: int = 1_000_000,
        n_updates: int = 10,
        callbacks: List["TrainCallback"] | None = None,
        window_size: int = 100,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        grad_clip: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        """
        Trains the agent on a set of vectorized Gymnasium environments using a
        `RolloutBuffer`.

        Parameters:
            envs (gym.vector.VectorEnv): the vectorized environments to train on
            batch_size (int): number of samples per mini-batch. Must compliment the `buffer_size`
            n_steps (int, optional): the total number of training steps
            n_updates (int, optional): the number of policy updates per batch
            callbacks (List[TrainCallback], optional): a list of training callbacks
                that are applied during the training process
            window_size (int, optional): controls the step rate for displaying
                information to the console and for calculating the reward moving
                average
            gamma (float, optional): the reward discount factor
            gae_lambda (float, optional): the GAE smoothing parameter
            clip_ratio (float, optional): the surrogate clipping ratio
            grad_clip (float): max norm gradient clip
            entropy_coef (float, optional): entropy exploration coefficient
        """
        if self.discrete:
            if not isinstance(envs.action_space, gym.spaces.Discrete):
                raise EnvironmentError(
                    f"Invalid '{envs.action_space=}'. Must be 'gym.spaces.Discrete' when 'discrete=True'."
                )
        else:
            if not isinstance(envs.action_space, gym.spaces.Box):
                raise EnvironmentError(
                    f"Invalid '{envs.action_space=}'. Must be 'gym.spaces.Box' when 'discrete=False'."
                )

        n_steps = closest_divisible(n_steps, batch_size)
        n_mini_batches = get_mini_batch_size(self.buffer_size, batch_size)
        total_updates = n_steps // batch_size

        # Add training details to config
        self.config = self.config.update(
            envs.spec.name,
            self._set_train_params(locals(), "rollout"),
        )

        # Display console details
        vec_training_info(
            self,
            envs.spec.id,
            envs.num_envs,
            n_steps,
            batch_size,
            n_mini_batches,
            window_size,
            callbacks or [],
            self.device,
        )

        with VecTrainHandler(
            self, envs, n_steps, batch_size, window_size, callbacks
        ) as handler:
            hidden = None
            critic_hidden = None

            states, _ = handler.envs.reset()
            current_step = 0

            for i_update in range(1, total_updates + 1):
                # Reset buffer at start of update
                self.buffer.empty()

                # Fill buffer
                while not self.buffer.is_full():
                    current_step += +1

                    with torch.no_grad():
                        actions, log_probs, _, hidden = self.actor(states, hidden)
                        values, critic_hidden = self.critic(states, critic_hidden)

                        next_states, rewards, terminated, truncated, _ = (
                            handler.envs.step(actions)
                        )
                        dones: torch.Tensor = terminated | truncated

                        if self.buffer.is_full():
                            break

                        self.buffer.add(
                            states,
                            actions,
                            rewards,
                            next_states,
                            dones,
                            log_probs,
                            values,
                        )

                        # Handle terminated episodes
                        if torch.any(dones):
                            mask = (~dones).float().unsqueeze(-1)
                            hidden = hidden * mask
                            critic_hidden = critic_hidden * mask

                        states = next_states

                # Buffer filled, update policy
                for i in range(n_updates):
                    self._anneal_lr(i, total_updates)
                    mini_batches = self.buffer.sample(batch_size)

                    for batch in mini_batches:
                        critic_loss, actor_loss = self._train_step(
                            batch,
                            gamma,
                            gae_lambda,
                            clip_ratio,
                            entropy_coef,
                            grad_clip,
                        )

                        reward = batch.rewards.mean()
                        handler.metrics.add_update(
                            i_update,
                            reward,
                            actor_loss,
                            critic_loss,
                        )
                        handler.update(i_update)

                if i_update % window_size == 0 or handler.stop():
                    handler.metrics.info(current_step, i_update)

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
        Makes an action prediction using the Actor network with exploration noise.

        Parameters:
            state (torch.Tensor): the current state
            hidden (torch.Tensor, optional): the current hidden state
            train_mode (bool, optional): whether to use stochastic actions

        Returns:
            action (torch.Tensor): the action prediction on the given state
            hidden (torch.Tensor): the Actor networks new hidden state
        """
        self.actor.eval() if not train_mode else self.actor.train()

        with torch.no_grad():
            state = state.unsqueeze(0) if state.dim() < 2 else state
            action, _, _, hidden = self.actor(state, hidden)

        return action, hidden
