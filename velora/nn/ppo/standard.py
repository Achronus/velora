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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import flatdim
from torch import nn, optim
from torch.distributions import Normal
from tqdm import tqdm

from velora.nn.buffer import RolloutBatch, RolloutBuffer
from velora.tracking.logger import MetricsLogger
from velora.utils.nn import set_torch_device
from velora.utils.transforms import to_torch_env


def layer_init(
    layer: nn.Linear,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear:
    """
    Initializes a linear layer in-place with orthogonal weights and
    constant bias.

    Parameters
    ----------
    layer : nn.Linear
        The linear layer to initialize
    std : float (optional)
        The gain (scaling factor) for the orthogonal weights.
        Default is `np.sqrt(2)`
    bias_const : float (optional)
        The constant value to fill the bias with. Default is `0.0`

    Returns
    -------
    layer : nn.Linear
        The initialized linear layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(
    env_id: str,
    num_envs: int = 1,
    *,
    gamma: float = 0.99,
    capture_video: bool = True,
    run_name: str = "",
) -> gym.vector.SyncVectorEnv:
    """
    Creates a set of vectorized environments with the standard PPO
    preprocessing wrappers applied to each one.

    Parameters
    ----------
    env_id : str
        The Gymnasium environment ID (e.g., `HalfCheetah-v4`)
    num_envs : int (optional)
        The number of parallel environments. Default is `1`
    gamma : float (optional)
        The discount factor for reward normalization. Default is `0.99`
    capture_video : bool (optional)
        Whether to record videos of the first environment.
        Default is `False`
    run_name : str (optional)
        The run name used for the video folder. Default is `""`

    Returns
    -------
    envs : gym.vector.SyncVectorEnv
        The vectorized environments
    """

    def thunk(idx: int):
        def _make() -> gym.Env:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"runs/videos/{run_name}")
            else:
                env = gym.make(env_id)

            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(np.asarray(obs), -10, 10),
                env.observation_space,
            )
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env,
                lambda reward: float(np.clip(float(reward), -10, 10)),
            )
            return env

        return _make

    return gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])


@dataclass
class PPOConfig:
    """
    A configuration for the PPO algorithm's hyperparameters.

    Parameters
    ----------
    total_timesteps : int (optional)
        The total number of environment timesteps to train for.
        Default is `1_000_000`
    lr : float (optional)
        The learning rate of the optimizer. Default is `3e-4`
    num_steps : int (optional)
        The number of steps to run in each environment per policy rollout.
        Default is `2048`
    anneal_lr : bool (optional)
        Whether to anneal the learning rate for the policy and value
        networks. Default is `True`
    gamma : float (optional)
        The discount factor. Default is `0.99`
    gae_lambda : float (optional)
        The lambda for the Generalized Advantage Estimation (GAE).
        Default is `0.95`
    num_minibatches : int (optional)
        The number of mini-batches per update. Default is `32`
    update_epochs : int (optional)
        The number of epochs (K) to update the policy. Default is `10`
    norm_adv : bool (optional)
        Whether to normalize the advantages. Default is `True`
    clip_coef : float (optional)
        The surrogate clipping coefficient. Default is `0.2`
    clip_vloss : bool (optional)
        Whether to use a clipped loss for the value function, as per the
        paper. Default is `True`
    ent_coef : float (optional)
        The entropy coefficient. Default is `0.0`
    vf_coef : float (optional)
        The value function coefficient. Default is `0.5`
    max_grad_norm : float (optional)
        The maximum norm for gradient clipping. Default is `0.5`
    target_kl : float (optional)
        The target KL divergence threshold. Default is `None`
    """

    total_timesteps: int = 1_000_000
    lr: float = 3e-4
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None


class MLPActorCritic(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) Actor-Critic from
    [CleanRL - PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy).

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_features, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(in_features, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, out_features), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, out_features))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the state-value estimate for a batch of observations.

        Parameters
        ----------
        x : torch.Tensor
            A batch of observations `(batch_size, in_features)`

        Returns
        -------
        value : torch.Tensor
            The critic's state-value estimates `(batch_size, 1)`
        """
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the actor's Gaussian policy and computes its
        log probability, entropy, and the critic's state-value estimate.

        Parameters
        ----------
        x : torch.Tensor
            A batch of observations `(batch_size, in_features)`
        action : torch.Tensor (optional)
            A batch of actions `(batch_size, out_features)` to evaluate
            instead of sampling new ones. Default is `None`

        Returns
        -------
        action : torch.Tensor
            The sampled (or given) actions `(batch_size, out_features)`
        log_prob : torch.Tensor
            The log probabilities of the actions, summed over the action
            dimensions `(batch_size,)`
        entropy : torch.Tensor
            The policy distribution's entropy, summed over the action
            dimensions `(batch_size,)`
        value : torch.Tensor
            The critic's state-value estimates `(batch_size, 1)`
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class PPO:
    """
    A Proximal Policy Optimization (PPO) agent for continuous action
    spaces, based on
    [CleanRL - PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy).

    Parameters
    ----------
    env_id : str
        The Gymnasium environment ID (e.g., `HalfCheetah-v4`)
    num_envs : int (optional)
        The number of parallel environments. Default is `1`
    seed : int (optional)
        Random number generator seed. Default is `28`
    config : PPOConfig (optional)
        The algorithm's hyperparameter configuration. When `None` uses
        `PPOConfig()` default values. Default is `None`
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
        *,
        num_envs: int = 1,
        seed: int = 28,
        config: PPOConfig | None = None,
        device: torch.device | None = None,
        project_name: str = "velora",
        exp_name: str = "cleanrl_ppo",
    ) -> None:
        config = config if config is not None else PPOConfig()
        self.exp_name = exp_name
        self.run_name = f"{env_id}_{exp_name}_{seed}_{int(time.time())}"
        envs = make_env(
            env_id,
            num_envs,
            gamma=config.gamma,
            run_name=self.run_name,
        )

        if not isinstance(envs.single_action_space, gym.spaces.Box):
            raise ValueError(
                "Invalid environment. Only `gym.spaces.Box` are supported."
            )

        self.config = config
        self.seed = seed
        self.device = device if device is not None else set_torch_device()
        self.envs = to_torch_env(envs, self.device)

        # Configure randomness
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Set training metrics
        self.batch_size = int(envs.num_envs * config.num_steps)
        self.minibatch_size = int(self.batch_size // config.num_minibatches)
        self.num_iterations = config.total_timesteps // self.batch_size

        self.global_step = 0

        # Setup agent
        self.agent = MLPActorCritic(
            flatdim(envs.single_observation_space),
            flatdim(envs.single_action_space),
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.lr, eps=1e-5)

        self.buffer = RolloutBuffer(
            self.envs.num_envs,
            self.config.num_steps,
            self.envs.single_observation_space.shape,  # type: ignore
            self.envs.single_action_space.shape,  # type: ignore
            device=self.device,
        )

        # Setup logging
        self.logger = MetricsLogger(
            "runs/cleanrl_ppo",
            project=project_name,
            run_name=self.run_name,
            group=f"{env_id}_{exp_name}",
            config={
                "env_id": env_id,
                "exp_name": exp_name,
                "num_envs": num_envs,
                "seed": seed,
                **asdict(config),
            },
        )

    def _set_anneal_lr(self, iteration: int) -> None:
        """
        Sets learning rate to be annealing, if enabled.

        Parameters
        ----------
        iteration : int
            Current training iteration
        """
        frac = 1.0 - (iteration - 1.0) / self.num_iterations
        lrnow = frac * self.config.lr
        self.optimizer.param_groups[0]["lr"] = lrnow

    def _collect(self, obs: torch.Tensor) -> Tuple[torch.Tensor, RolloutBatch]:
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

        for _ in range(0, self.config.num_steps):
            self.global_step += self.envs.num_envs

            with torch.no_grad():
                actions, log_probs, _, values = self.agent.get_action_and_value(obs)
                values = values.flatten()

            next_obs, rewards, terminations, truncations, info = self.envs.step(
                actions.cpu()
            )
            dones = torch.logical_or(terminations, truncations)

            # Log completed episode stats
            if "episode" in info:
                episode = info["episode"]
                mask = torch.as_tensor(info["_episode"])

                for idx in mask.nonzero().flatten().tolist():
                    self.logger.log(
                        "charts",
                        self.global_step,
                        {
                            "episodic_return": float(episode["r"][idx]),
                            "episodic_length": float(episode["l"][idx]),
                        },
                    )

            # Store in buffer
            self.buffer.add(obs, actions, log_probs, rewards.view(-1), dones, values)
            obs = next_obs.float()

        return obs, self.buffer.get()

    def _compute_policy_loss(
        self,
        advantages: torch.Tensor,
        ratio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes a mini-batches policy gradient loss.

        Parameters
        ----------
        advantages : torch.Tensor
            Current mini-batch of advantages
        ratio : torch.Tensor
            The probability ratios between the updated and rollout
            policies, `pi_new(a|s) / pi_old(a|s)`, computed as
            `(new_log_probs - old_log_probs).exp()`

        Returns
        -------
        pg_loss : torch.Tensor
            Policy gradient loss
        """
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1 - self.config.clip_coef,
            1 + self.config.clip_coef,
        )
        return torch.max(pg_loss1, pg_loss2).mean()

    def _compute_value_loss(
        self,
        new_values: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes a mini-batches value loss.

        Parameters
        ----------
        new_values : torch.Tensor
            Mini-batch of next values
        returns : torch.Tensor
            Mini-batch of returns
        values : torch.Tensor
            Mini-batch of current values

        Returns
        -------
        v_loss : torch.Tensor
            Value loss
        """
        new_values = new_values.view(-1)

        if self.config.clip_vloss:
            v_loss_unclipped = (new_values - returns) ** 2
            v_clipped = values + torch.clamp(
                new_values - values,
                -self.config.clip_coef,
                self.config.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            return 0.5 * v_loss_max.mean()

        return 0.5 * ((new_values - returns) ** 2).mean()

    def _update_minibatch(
        self,
        rollout: RolloutBatch,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        indices: np.ndarray,
    ) -> Dict[str, float]:
        """
        Performs a single gradient update on a mini-batch of experience.

        Parameters
        ----------
        rollout : RolloutBatch
            Flattened batch of rollout experience `(batch_size, ...)`
        advantages : torch.Tensor
            The GAE advantage estimates `(batch_size,)`
        returns : torch.Tensor
            The discounted returns `(batch_size,)`
        indices : np.ndarray
            The mini-batch sample indices `(minibatch_size,)`

        Returns
        -------
        stats : Dict[str, float]
            The mini-batch's loss metric keys:
            [`value`, `policy`, `entropy`,
            `total`, `old_approx_kl`, `approx_kl`, `clip_frac`]
        """
        _, new_log_probs, entropy, new_values = self.agent.get_action_and_value(
            rollout.obs[indices],
            rollout.actions[indices],
        )
        log_ratio = new_log_probs - rollout.log_probs[indices]
        ratio = log_ratio.exp()

        with torch.no_grad():
            old_approx_kl = (-log_ratio).mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_frac = (
                ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
            )

        # Normalize advantages
        minibatch_advantages = advantages[indices]

        if self.config.norm_adv:
            minibatch_advantages = (
                minibatch_advantages - minibatch_advantages.mean()
            ) / (minibatch_advantages.std() + 1e-8)

        # Compute losses
        pg_loss = self._compute_policy_loss(minibatch_advantages, ratio)
        v_loss = self._compute_value_loss(
            new_values,
            returns[indices],
            rollout.values[indices],
        )
        entropy_loss = entropy.mean()
        loss = (
            pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
        )

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "value": v_loss.item(),
            "policy": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "total": loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clip_frac": clip_frac,
        }

    def train(self) -> None:
        """Trains the agent."""
        obs, _ = self.envs.reset(seed=self.seed)
        obs: torch.Tensor = obs.float()

        progress = tqdm(
            total=self.num_iterations * self.batch_size,
            desc="Training",
            unit="step",
        )

        for iteration in range(1, self.num_iterations + 1):
            if self.config.anneal_lr:
                self._set_anneal_lr(iteration)

            obs, rollout = self._collect(obs)

            with torch.no_grad():
                last_value = self.agent.get_value(obs)

            advantages, returns = self.buffer.compute_returns_and_advantage(
                last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

            # Optimize the agent networks
            batch_indices = np.arange(self.batch_size)
            stats: Dict[str, float] = {}
            clip_fracs = []

            # Flatten for mini-batching
            rollout = rollout.flatten()
            advantages = advantages.flatten(0, 1)
            returns = returns.flatten(0, 1)

            # Start policy update epochs
            for _ in range(self.config.update_epochs):
                np.random.shuffle(batch_indices)

                # Start mini-batch iterations
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    stats = self._update_minibatch(
                        rollout,
                        advantages,
                        returns,
                        batch_indices[start:end],
                    )
                    clip_fracs.append(stats["clip_frac"])

                # Target exceeded, next iteration
                if (
                    self.config.target_kl is not None
                    and stats["approx_kl"] > self.config.target_kl
                ):
                    break

            # Compute variance for logging
            y_pred, y_true = rollout.values, returns
            var_y = torch.var(y_true)
            explained_var = float(
                torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y
            )

            self.logger.log(
                "losses",
                self.global_step,
                {
                    **stats,
                    "clip_frac": np.mean(clip_fracs).item(),
                    "explained_variance": explained_var,
                },
            )

            # Update progress bar
            progress.update(self.batch_size)
            progress.set_postfix(
                loss=f"{stats['total']:.3f}",
                approx_kl=f"{stats['approx_kl']:.4f}",
            )

        # End of training
        progress.close()
        self.envs.close()

        # Upload recorded videos
        video_dir = Path("runs", "videos") / self.run_name
        videos = sorted(video_dir.glob("*.mp4"))

        if videos:
            self.logger.log_video("runs/videos/episodes", self.global_step, videos)  # type: ignore

        self.logger.close()
