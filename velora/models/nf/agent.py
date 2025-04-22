from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Self, Tuple, Type, Union

from velora.callbacks import TrainCallback
from velora.training.display import training_info
from velora.training.handler import TrainHandler

try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # pragma: no cover

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from velora.gym.wrap import DiscreteGymNames, ContinuousGymNames  # pragma: no cover
    from velora.buffer.experience import BatchExperience  # pragma: no cover

from velora.buffer.replay import ReplayBuffer
from velora.models.base import RLModuleAgent
from velora.models.config import ModelDetails, RLAgentConfig, TorchConfig
from velora.models.nf.modules import (
    ActorModule,
    ActorModuleDiscrete,
    CriticModule,
    CriticModuleDiscrete,
    EntropyModule,
    EntropyModuleDiscrete,
)
from velora.utils.restore import load_model, save_model

StateDictKeys = Literal["modules", "optimizers"]


class NeuroFlowCT(RLModuleAgent):
    """
    ???+ abstract "Documentation"

        > [User Guide - Tutorials: NeuroFlow - Continuous](https://velora.achronus.dev/learn/tutorial/agents/nf)

    A custom Liquid RL agent that combines a variety of RL techniques.

    Designed for `continuous` action spaces.
    """

    def __init__(
        self,
        env_id: Union[str, "ContinuousGymNames"],
        actor_neurons: int,
        critic_neurons: int,
        *,
        optim: Type[optim.Optimizer] = optim.Adam,
        buffer_size: int = 1_000_000,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        initial_alpha: float = 1.0,
        log_std: Tuple[float, float] = (-5, 2),
        tau: float = 0.005,
        gamma: float = 0.99,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters:
            env_id (str): the Gymnasium environment ID to train the model on
            actor_neurons (int): number of decision nodes (inter and command nodes)
                for the actor
            critic_neurons (int): number of decision nodes (inter and command nodes)
                for the critic
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
            tau (float, optional): the soft update factor used to slowly update
                the target networks
            gamma (float, optional): the reward discount factor
            device (torch.device, optional): the device to perform computations on
            seed (int, optional): random number seed for experiment
                reproducibility. When `None` generates a seed automatically
        """
        env = gym.make(env_id, render_mode="rgb_array")

        if not isinstance(env.action_space, gym.spaces.Box):
            raise EnvironmentError(
                f"Invalid '{env.action_space=}'. Must be 'gym.spaces.Box'."
            )

        super().__init__(
            env,
            actor_neurons,
            critic_neurons,
            buffer_size,
            optim,
            device,
            seed,
        )

        self.initial_alpha = initial_alpha
        self.log_std = log_std
        self.gamma = gamma
        self.tau = tau

        self.actor: ActorModule = ActorModule(
            self.state_dim,
            self.actor_neurons,
            self.action_dim,
            self.action_scale,
            self.action_bias,
            log_std_min=log_std[0],
            log_std_max=log_std[1],
            optim=optim,
            lr=actor_lr,
            device=self.device,
        )

        self.critic: CriticModule = CriticModule(
            self.state_dim,
            self.critic_neurons,
            self.action_dim,
            optim=optim,
            lr=critic_lr,
            tau=tau,
            device=self.device,
        )

        self.hidden_dim = self.actor.hidden_size

        self.entropy: EntropyModule = EntropyModule(
            self.action_dim,
            initial_alpha=initial_alpha,
            optim=optim,
            lr=alpha_lr,
            device=device,
        )

        self.loss = nn.MSELoss()
        self.buffer: ReplayBuffer = ReplayBuffer(
            buffer_size,
            self.state_dim,
            self.action_dim,
            self.actor.hidden_size,
            device=device,
        )

        self.active_params = self.actor.active_params + self.critic.active_params
        self.total_params = self.actor.total_params + self.critic.total_params

        # Init config details
        self.config = RLAgentConfig(
            agent=self.__class__.__name__,
            env=env_id,
            seed=self.seed,
            model_details=ModelDetails(
                **locals(),
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                exploration_type="Entropy",
                actor=self.actor.config,
                critic=self.critic.config,
                entropy=self.entropy.config(),
            ),
            buffer=self.buffer.config(),
            torch=TorchConfig(
                device=str(self.device),
                optimizer=optim.__name__,
                loss=self.loss.__class__.__name__,
            ),
        )

        self.metadata = self.set_metadata(locals(), self.seed)

    def _update_critics(self, batch: "BatchExperience") -> torch.Tensor:
        """
        Helper method. Performs Critic network updates.

        Parameters:
            batch (BatchExperience): an object containing a batch of experience
                with `(states, actions, rewards, next_states, dones, hidden)`
                from the buffer

        Returns:
            critic_loss (torch.Tensor): total Critic network loss `(c1_loss + c2_loss)`.
        """
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.forward(
                batch.next_states, batch.hiddens
            )

            # Compute target Q-value
            next_q = self.critic.target_predict(batch.next_states, next_actions)
            next_q = next_q - self.entropy.alpha * next_log_probs
            target_q = batch.rewards + (1 - batch.dones) * self.gamma * next_q

        current_q1, current_q2 = self.critic.predict(batch.states, batch.actions)

        # Calculate loss
        c1_loss: torch.Tensor = self.loss(current_q1, target_q)
        c2_loss: torch.Tensor = self.loss(current_q2, target_q)
        critic_loss: torch.Tensor = c1_loss + c2_loss

        # Update critics
        self.critic.gradient_step(c1_loss, c2_loss)

        return critic_loss

    def _train_step(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Helper method. Performs a single training step.

        Parameters:
            batch_size (int): number of samples in a batch

        Returns:
            losses (Dict[str, torch.Tensor]): a dictionary of losses -

            - critic: the total critic loss.
            - actor: the actor loss.
            - entropy: the entropy loss.
        """
        batch = self.buffer.sample(batch_size)

        # Compute critic loss
        critic_loss = self._update_critics(batch)

        # Make predictions
        actions, log_probs, _ = self.actor.forward(batch.states, batch.hiddens)
        q1, q2 = self.critic.predict(batch.states, actions)

        # Compute actor and entropy losses
        next_q = torch.min(q1, q2)
        actor_loss = (self.entropy.alpha * log_probs - next_q).mean()
        entropy_loss = self.entropy.compute_loss(log_probs)

        # Update gradients
        self.actor.gradient_step(actor_loss)
        self.entropy.gradient_step(entropy_loss)

        # Update target networks
        self.critic.update_targets()

        return {
            "critic": critic_loss.detach(),
            "actor": actor_loss.detach(),
            "entropy": entropy_loss.detach(),
        }

    @override
    def train(
        self,
        batch_size: int,
        *,
        n_episodes: int = 10_000,
        callbacks: List["TrainCallback"] | None = None,
        log_freq: int = 10,
        display_count: int = 100,
        window_size: int = 100,
        max_steps: int = 1000,
        warmup_steps: int | None = None,
    ) -> None:
        """
        Trains the agent on a Gymnasium environment using a `ReplayBuffer`.

        Parameters:
            batch_size (int): the number of features in a single batch
            n_episodes (int, optional): the total number of episodes to train for
            callbacks (List[TrainCallback], optional): a list of training callbacks
                that are applied during the training process
            log_freq (int, optional): metric logging frequency (in episodes)
            display_count (int, optional): console training progress frequency
                (in episodes)
            window_size (int, optional): the reward moving average size
                (in episodes)
            max_steps (int, optional): the total number of steps per episode
            warmup_steps (int, optional): the number of samples to generate in the
                buffer before starting training. If `None` uses `batch_size * 2`
        """
        warmup_steps = batch_size * 2 if warmup_steps is None else warmup_steps

        # Add training details to config
        self.config = self.config.update(self._set_train_params(locals()))

        # Display console details
        self.env.reset(seed=self.seed)  # Set seed
        training_info(
            self,
            n_episodes,
            batch_size,
            window_size,
            warmup_steps,
            callbacks or [],
        )

        if warmup_steps > 0:
            self.buffer.warm(self, warmup_steps, 2 if warmup_steps < 8 else 8)

        with TrainHandler(
            self, n_episodes, max_steps, log_freq, window_size, callbacks
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

                    losses = self._train_step(batch_size)

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

                if current_ep % log_freq == 0 or current_ep == n_episodes:
                    handler.log(current_ep, "episode")

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
        self.actor.eval_mode()
        with torch.no_grad():
            state = state.unsqueeze(0) if state.dim() < 2 else state

            if not train_mode:
                action, hidden = self.actor.predict(state, hidden)
            else:
                action, _, hidden = self.actor.forward(state, hidden)

        self.actor.train_mode()
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
            + ", ".join([f"{key}={val}" for key, val in self.metadata.items()])
            + ")"
        )


class NeuroFlow(RLModuleAgent):
    """
    ???+ abstract "Documentation"

        > [User Guide - Tutorials: NeuroFlow - Discrete](https://velora.achronus.dev/learn/tutorial/agents/nf2)

    A custom Liquid RL agent that combines a variety of RL techniques.

    Designed for `discrete` action spaces.
    """

    def __init__(
        self,
        env_id: Union[str, "DiscreteGymNames"],
        actor_neurons: int,
        critic_neurons: int,
        *,
        optim: Type[optim.Optimizer] = optim.Adam,
        buffer_size: int = 1_000_000,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        initial_alpha: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters:
            env_id (str): the Gymnasium environment ID to train the model on
            actor_neurons (int): number of decision nodes (inter and command nodes)
                for the actor
            critic_neurons (int): number of decision nodes (inter and command nodes)
                for the critic
            optim (Type[torch.optim.Optimizer], optional): the type of `PyTorch`
                optimizer to use
            buffer_size (int, optional): the maximum size of the `ReplayBuffer`
            actor_lr (float, optional): the actor optimizer learning rate
            critic_lr (float, optional): the critic optimizer learning rate
            alpha_lr (float, optional): the entropy parameter learning rate
            initial_alpha (float, optional): the starting entropy coefficient value
            tau (float, optional): the soft update factor used to slowly update
                the target networks
            gamma (float, optional): the reward discount factor
            device (torch.device, optional): the device to perform computations on
            seed (int, optional): random number seed for experiment
                reproducibility. When `None` generates a seed automatically
        """
        env = gym.make(env_id, render_mode="rgb_array")

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise EnvironmentError(
                f"Invalid '{env.action_space=}'. Must be 'gym.spaces.Discrete'."
            )

        super().__init__(
            env,
            actor_neurons,
            critic_neurons,
            buffer_size,
            optim,
            device,
            seed,
        )

        self.initial_alpha = initial_alpha
        self.gamma = gamma
        self.tau = tau

        self.actor: ActorModuleDiscrete = ActorModuleDiscrete(
            self.state_dim,
            self.actor_neurons,
            self.action_dim,
            optim=optim,
            lr=actor_lr,
            device=self.device,
        )

        self.critic: CriticModuleDiscrete = CriticModuleDiscrete(
            self.state_dim,
            self.critic_neurons,
            self.action_dim,
            optim=optim,
            lr=critic_lr,
            tau=tau,
            device=self.device,
        )

        self.hidden_dim = self.actor.hidden_size

        self.entropy: EntropyModuleDiscrete = EntropyModuleDiscrete(
            self.action_dim,
            initial_alpha=initial_alpha,
            optim=optim,
            lr=alpha_lr,
            device=device,
        )

        self.loss = nn.MSELoss()
        self.buffer: ReplayBuffer = ReplayBuffer(
            buffer_size,
            self.state_dim,
            1,
            self.actor.hidden_size,
            device=device,
        )

        self.active_params = self.actor.active_params + self.critic.active_params
        self.total_params = self.actor.total_params + self.critic.total_params

        # Init config details
        self.config = RLAgentConfig(
            agent=self.__class__.__name__,
            env=env_id,
            seed=self.seed,
            model_details=ModelDetails(
                **locals(),
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                action_type="discrete",
                exploration_type="Entropy",
                actor=self.actor.config,
                critic=self.critic.config,
                entropy=self.entropy.config(),
            ),
            buffer=self.buffer.config(),
            torch=TorchConfig(
                device=str(self.device),
                optimizer=optim.__name__,
                loss=self.loss.__class__.__name__,
            ),
        )

        self.metadata = self.set_metadata(locals(), self.seed)

    def _update_critics(self, batch: "BatchExperience") -> torch.Tensor:
        """
        Helper method. Performs Critic network updates.

        Parameters:
            batch (BatchExperience): an object containing a batch of experience
                with `(states, actions, rewards, next_states, dones, hidden)`
                from the buffer

        Returns:
            critic_loss (torch.Tensor): total Critic network loss `(c1_loss + c2_loss)`.
        """
        with torch.no_grad():
            _, next_probs, _, _ = self.actor.forward(batch.next_states, batch.hiddens)
            next_log_probs = torch.log(next_probs + 1e-8)

            # Compute target Q-value (all actions)
            min_next_q = self.critic.target_predict(batch.next_states)  # [b, a_dim]

            next_q = next_probs * (min_next_q - self.entropy.alpha * next_log_probs)
            next_q = torch.sum(next_q, dim=-1, keepdim=True)
            target_q = batch.rewards + (1 - batch.dones) * self.gamma * next_q  # [b, 1]

        # Compute Q-value predictions for current critics (all actions)
        current_q1, current_q2 = self.critic.predict(batch.states)  # [b, a_dim]

        # Select Q-values for the actions taken in the batch - shape: (b, 1)
        current_q1 = current_q1.gather(1, batch.actions.long())
        current_q2 = current_q2.gather(1, batch.actions.long())

        # Calculate loss
        c1_loss: torch.Tensor = self.loss(current_q1, target_q)
        c2_loss: torch.Tensor = self.loss(current_q2, target_q)
        critic_loss: torch.Tensor = c1_loss + c2_loss

        # Update critics
        self.critic.gradient_step(c1_loss, c2_loss)

        return critic_loss

    def _train_step(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Helper method. Performs a single training step.

        Parameters:
            batch_size (int): number of samples in a batch

        Returns:
            losses (Dict[str, torch.Tensor]): a dictionary of losses -

            - critic: the total critic loss.
            - actor: the actor loss.
            - entropy: the entropy loss.
        """
        batch = self.buffer.sample(batch_size)

        # Compute critic loss
        critic_loss = self._update_critics(batch)

        # Make predictions
        _, probs, log_probs, _ = self.actor.forward(batch.states, batch.hiddens)
        q1, q2 = self.critic.predict(batch.states)

        # Compute actor and entropy losses
        next_q = torch.min(q1, q2)  # [b, a_dim]
        actor_loss = probs * (self.entropy.alpha * log_probs - next_q)
        actor_loss = torch.sum(actor_loss, dim=-1, keepdim=False).mean()

        entropy_loss = self.entropy.compute_loss(
            probs,
            torch.log(probs + 1e-8),
        )

        # Update gradients
        self.actor.gradient_step(actor_loss)
        self.entropy.gradient_step(entropy_loss)

        # Update target networks
        self.critic.update_targets()

        return {
            "critic": critic_loss.detach(),
            "actor": actor_loss.detach(),
            "entropy": entropy_loss.detach(),
        }

    @override
    def train(
        self,
        batch_size: int,
        *,
        n_episodes: int = 10_000,
        callbacks: List["TrainCallback"] | None = None,
        log_freq: int = 10,
        display_count: int = 100,
        window_size: int = 100,
        max_steps: int = 1000,
        warmup_steps: int | None = None,
    ) -> None:
        """
        Trains the agent on a Gymnasium environment using a `ReplayBuffer`.

        Parameters:
            batch_size (int): the number of features in a single batch
            n_episodes (int, optional): the total number of episodes to train for
            callbacks (List[TrainCallback], optional): a list of training callbacks
                that are applied during the training process
            log_freq (int, optional): metric logging frequency (in episodes)
            display_count (int, optional): console training progress frequency
                (in episodes)
            window_size (int, optional): the reward moving average size
                (in episodes)
            max_steps (int, optional): the total number of steps per episode
            warmup_steps (int, optional): the number of samples to generate in the
                buffer before starting training. If `None` uses `batch_size * 2`
        """
        warmup_steps = batch_size * 2 if warmup_steps is None else warmup_steps

        # Add training details to config
        self.config = self.config.update(self._set_train_params(locals()))

        # Display console details
        self.env.reset(seed=self.seed)  # Set seed
        training_info(
            self,
            n_episodes,
            batch_size,
            window_size,
            warmup_steps,
            callbacks or [],
        )

        if warmup_steps > 0:
            self.buffer.warm(self, warmup_steps, 2 if warmup_steps < 8 else 8)

        with TrainHandler(
            self, n_episodes, max_steps, log_freq, window_size, callbacks
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

                    losses = self._train_step(batch_size)

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

                if current_ep % log_freq == 0 or current_ep == n_episodes:
                    handler.log(current_ep, "episode")

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
        self.actor.eval_mode()
        with torch.no_grad():
            state = state.unsqueeze(0) if state.dim() < 2 else state

            if not train_mode:
                action, hidden = self.actor.predict(state, hidden)
            else:
                action, _, _, hidden = self.actor.forward(state, hidden)

        self.actor.train_mode()
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
            + ", ".join([f"{key}={val}" for key, val in self.metadata.items()])
            + ")"
        )
