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

import random
from collections.abc import Callable
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import flatdim
from torch import nn, optim

from velora.nn.buffer import RolloutBatch, RolloutBuffer
from velora.nn.ppo.config import PPOConfig
from velora.nn.ppo.loss import PPOLoss
from velora.nn.ppo.networks import MLPActorCritic
from velora.nn.ppo.sampler import PPOActionSampler
from velora.nn.ppo.utils import compute_gae, make_env, normalize_advantages
from velora.tracking.run import RunTracker, RunTrackerConfig
from velora.utils.diagnostics import PPODiagnostics
from velora.utils.loader import MiniBatchData, MiniBatchLoader
from velora.utils.nn import set_torch_device
from velora.utils.transforms import to_torch_env


class PPO:
    """
    A Proximal Policy Optimization (PPO) trainer for continuous action
    spaces, based on
    [CleanRL - PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy).

    Acts as the base trainer for all PPO variants. Core logic is split
    into small utility methods that subclasses override to change
    behaviour (e.g., `RPO` and `LSTMPPO`).

    Parameters
    ----------
    env_id : str
        The Gymnasium environment ID (e.g., `HalfCheetah-v4`)
    config : PPOConfig
        The algorithm's hyperparameter configuration
    num_envs : int (optional)
        The number of parallel environments. Default is `1`
    seed : int (optional)
        Random number generator seed. Default is `28`
    agent : Callable[..., nn.Module] (optional)
        The actor-critic network class (or factory) to use. Called with
        `(in_features, out_features)`. Must satisfy the `ActorCritic`
        contract. When `None` uses `MLPActorCritic`. Default is `None`
    device : torch.device (optional)
        Device to load tensors onto. When `None`, sets device to CUDA
        or CPU automatically. Default is `None`
    project_name : str (optional)
        The project name for wandb metric logging. Default is `velora`
    exp_name : str (optional)
        The experiment name, shared by all seed runs of the same
        algorithm variant. Used for run grouping and benchmark tooling.
        Default is `ppo`
    """

    def __init__(
        self,
        env_id: str,
        config: PPOConfig,
        *,
        num_envs: int = 1,
        seed: int = 28,
        agent: Callable[..., nn.Module] | None = None,
        device: torch.device | None = None,
        project_name: str = "velora",
        exp_name: str = "ppo",
    ) -> None:
        agent_cls = agent if agent is not None else MLPActorCritic

        # Set training metrics
        self.batch_size = int(num_envs * config.num_steps)
        self.minibatch_size = int(self.batch_size // config.num_minibatches)
        self.num_iterations = config.total_timesteps // self.batch_size

        # Configure tracker
        self.tracker = RunTracker(
            self.num_iterations * self.batch_size,
            RunTrackerConfig(
                env_id,
                exp_name,
                project_name,
                seed,
            ),
            metadata={
                "num_envs": num_envs,
                "agent": agent_cls.__name__,
                **asdict(config),
            },
        )

        # Create environments
        envs = make_env(
            env_id,
            num_envs,
            gamma=config.gamma,
            run_name=self.tracker.config.run_name,
        )

        if not isinstance(envs.single_action_space, gym.spaces.Box):
            raise TypeError("Invalid environment. Only `gym.spaces.Box` are supported.")

        self.config = config
        self.seed = seed
        self.device = device if device is not None else set_torch_device()
        self.envs = to_torch_env(envs, self.device)

        # Configure randomness
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Setup agent
        self.agent: nn.Module = agent_cls(
            flatdim(envs.single_observation_space),
            flatdim(envs.single_action_space),
        ).to(self.device)
        self._compile_agent()

        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.lr, eps=1e-5)
        self.scheduler = (
            optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.num_iterations,
            )
            if self.config.anneal_lr
            else optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        )

        self.buffer = RolloutBuffer(
            self.envs.num_envs,
            self.config.num_steps,
            self.envs.single_observation_space.shape,  # type: ignore
            self.envs.single_action_space.shape,  # type: ignore
            device=self.device,
        )

        self.loss = PPOLoss(
            self.config.clip_coef,
            self.config.clip_vloss,
            self.config.ent_coef,
            self.config.vf_coef,
        )
        self.sampler = PPOActionSampler()
        self.mb_loader = MiniBatchLoader(self.batch_size, self.minibatch_size)
        self.diagnostics = PPODiagnostics(self.config.clip_coef)

        self._next_done = torch.zeros(self.envs.num_envs, device=self.device)

    def _compile_agent(self) -> None:
        """Compiles the agent network with `torch.compile`."""
        self.agent.compile()

    def _on_rollout_start(self) -> None:
        """
        Hook called before each rollout begins. Override point for
        variants that track per-rollout state (e.g., `LSTMPPO`).
        """

    def _on_rollout_end(self) -> None:
        """
        Hook called after each rollout completes. Override point for
        variants that track per-rollout state (e.g., `LSTMPPO`).
        """

    def _rollout_step(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects actions for a single environment step during rollouts.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of observations `(num_envs, in_features)`

        Returns
        -------
        actions : torch.Tensor
            The sampled actions `(num_envs, out_features)`
        log_probs : torch.Tensor
            The log probabilities of the actions `(num_envs,)`
        values : torch.Tensor
            The critic's state-value estimates `(num_envs,)`
        """
        with torch.no_grad():
            mean, log_std, values = self.agent(obs)
            actions, log_probs = self.sampler.sample(mean, log_std)

        return actions, log_probs, values.flatten()

    def _bootstrap_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes the value estimate for the observation following the
        rollout's final step, used to bootstrap the GAE.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of observations `(num_envs, in_features)`

        Returns
        -------
        value : torch.Tensor
            The critic's state-value estimates `(num_envs, 1)`
        """
        with torch.no_grad():
            return self.agent(obs)[2]

    def _collect(self, obs: torch.Tensor) -> tuple[torch.Tensor, RolloutBatch]:
        """
        Collects samples of experience from the environments and returns
        the filled buffer.

        Parameters
        ----------
        obs : torch.Tensor
            Latest observation

        Returns
        -------
        next_obs : torch.Tensor
            Next observation
        rollout : RolloutBatch
            Full batch of rollout experience
        """
        self.buffer.reset()
        self._on_rollout_start()

        # Collect experience and store in buffer
        for _ in range(self.config.num_steps):
            self.tracker.advance(self.envs.num_envs)

            actions, log_probs, values = self._rollout_step(obs)

            next_obs, rewards, terminations, truncations, info = self.envs.step(
                actions.cpu()
            )
            dones = torch.logical_or(terminations, truncations)
            self._next_done = dones.float()

            # Log completed episode stats
            self.tracker.log_episodic(info)

            # Store in buffer
            self.buffer.add(obs, actions, log_probs, rewards.view(-1), dones, values)
            obs = next_obs.float()

        self._on_rollout_end()
        return obs, self.buffer.get()

    def _evaluate(
        self,
        minibatch: MiniBatchData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluates a mini-batch of rollout actions under the current
        policy. Override point for variants that change how actions
        are re-evaluated (e.g., `LSTMPPO`).

        Parameters
        ----------
        minibatch : MiniBatchData
            A mini-batch sample of rollouts

        Returns
        -------
        new_log_probs : torch.Tensor
            The log probabilities of the actions under the current
            policy `(minibatch_size,)`
        entropy : torch.Tensor
            The policy distribution's entropy `(minibatch_size,)`
        new_values : torch.Tensor
            The critic's state-value estimates `(minibatch_size, 1)`
        """
        mean, log_std, new_values = self.agent(minibatch.obs)
        log_probs, entropy = self.sampler.evaluate(
            mean,
            log_std,
            actions=minibatch.actions,
        )
        return log_probs, entropy, new_values

    def _update_minibatch(self, minibatch: MiniBatchData) -> None:
        """
        Performs a single gradient update on a mini-batch of experience.

        Parameters
        ----------
        minibatch : MiniBatchData
            A mini-batch sample of rollouts
        """
        new_log_probs, entropy, new_values = self._evaluate(minibatch)
        log_ratio = new_log_probs - minibatch.log_probs
        ratio = log_ratio.exp()

        # Handle advantages
        advantages = (
            normalize_advantages(minibatch.advantages)
            if self.config.norm_adv
            else minibatch.advantages
        )

        # Compute loss
        pg_loss = self.loss.policy(advantages, ratio)
        v_loss = self.loss.value(new_values, minibatch.returns, minibatch.values)
        entropy_loss = self.loss.entropy(entropy)
        total_loss = self.loss.total(pg_loss, v_loss, entropy_loss)

        # Add stats to diagnostics
        self.diagnostics.add_surrogate(log_ratio, ratio)
        self.diagnostics.add_losses(pg_loss, v_loss, entropy_loss, total_loss)

        # Backpropagate
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

    def train(self) -> None:
        """Trains the agent."""
        obs, _ = self.envs.reset(seed=self.seed)
        obs: torch.Tensor = obs.float()

        with self.tracker as tracker:
            for _ in range(1, self.num_iterations + 1):
                obs, rollout = self._collect(obs)
                last_value = self._bootstrap_value(obs)

                advantages, returns = compute_gae(
                    rollout.rewards,
                    rollout.values,
                    rollout.dones,
                    last_value,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )

                # Flatten for mini-batching
                rollout = rollout.flatten()
                advantages = advantages.flatten(0, 1)
                returns = returns.flatten(0, 1)

                self.mb_loader.load(rollout, advantages, returns)

                # Start policy update epochs
                for _ in range(self.config.update_epochs):
                    for minibatch in self.mb_loader:
                        self._update_minibatch(minibatch)

                    # Target exceeded, next iteration
                    if (
                        self.config.target_kl is not None
                        and self.diagnostics.last_approx_kl > self.config.target_kl
                    ):
                        break

                # Anneal LR (if enabled)
                self.scheduler.step()

                # Log metrics
                stats = self.diagnostics.summary(rollout.values, returns)
                tracker.log(
                    "losses",
                    metrics=asdict(stats),
                    bar_details={
                        "loss": f"{stats.total:.4f}",
                        "approx_kl": f"{stats.approx_kl:.4f}",
                    },
                )

        # End of training
        self.envs.close()
