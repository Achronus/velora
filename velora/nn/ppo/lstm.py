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

from collections.abc import Callable

import torch
from torch import nn

from velora.nn.ppo.config import PPOConfig
from velora.nn.ppo.networks import LSTMActorCritic
from velora.nn.ppo.standard import PPO
from velora.utils.loader import LSTMMiniBatchData, LSTMMiniBatchLoader


class LSTMPPO(PPO):
    """
    A recurrent Proximal Policy Optimization (PPO) trainer for
    continuous action spaces, adapted from
    [CleanRL - PPO LSTM](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy).

    Extends `PPO` by threading a recurrent state through rollouts and
    policy updates, and by sampling mini-batches per-environment so
    each trajectory's timesteps stay in order for the recurrent state
    to replay correctly.

    The agent network is not compiled - the per-timestep loop causes
    `torch.compile` graph breaks between the rollout (`T=1`) and update
    (`T=num_steps`) call shapes. Compiling the encoder and heads only
    is a possible future optimization.

    Parameters
    ----------
    env_id : str
        The Gymnasium environment ID (e.g., `HalfCheetah-v4`)
    config : PPOConfig
        The algorithm's hyperparameter configuration
    num_envs : int (optional)
        The number of parallel environments. Must be divisible by
        `config.num_minibatches`. Default is `1`
    seed : int (optional)
        Random number generator seed. Default is `28`
    agent : Callable[..., nn.Module] (optional)
        The actor-critic network class (or factory) to use. Called with
        `(in_features, out_features)`. Must satisfy the
        `RecurrentActorCritic` contract. When `None` uses
        `LSTMActorCritic`. Default is `None`
    device : torch.device (optional)
        Device to load tensors onto. When `None`, sets device to CUDA
        or CPU automatically. Default is `None`
    project_name : str (optional)
        The project name for wandb metric logging. Default is `velora`
    exp_name : str (optional)
        The experiment name, shared by all seed runs of the same
        algorithm variant. Used for run grouping and benchmark tooling.
        Default is `lstm_ppo`
    """

    mb_loader: LSTMMiniBatchLoader

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
        exp_name: str = "lstm_ppo",
    ) -> None:
        agent = agent if agent is not None else LSTMActorCritic

        super().__init__(
            env_id,
            config,
            num_envs=num_envs,
            seed=seed,
            agent=agent,
            device=device,
            project_name=project_name,
            exp_name=exp_name,
        )

        # Override defaults
        self.mb_loader = LSTMMiniBatchLoader(
            config.num_steps,
            self.envs.num_envs,
            config.num_minibatches,
        )

        self._next_state = self.agent.init_state(self.envs.num_envs, self.device)  # type: ignore

    def _compile_agent(self) -> None:
        """
        Skips compilation. The recurrent per-timestep loop is not
        `torch.compile` friendly.
        """

    def _on_rollout_start(self) -> None:
        """
        Snapshots the recurrent state and completion flags entering the
        rollout, for replaying trajectories during policy updates.
        """
        self._initial_state = (
            self._next_state[0].clone(),
            self._next_state[1].clone(),
        )
        self._initial_done = self._next_done.clone()

    def _on_rollout_end(self) -> None:
        """
        Builds the completion flags entering each rollout timestep and
        stores them in the mini-batch loader, along with the recurrent
        state entering the rollout.

        The buffer stores `dones[t]` as the flags *after* step `t`, so
        the flags entering step `t` are the initial flags followed by
        `dones[:-1]`.
        """
        update_dones = torch.cat(
            [self._initial_done.unsqueeze(0), self.buffer.dones[:-1]],
            dim=0,
        ).flatten(0, 1)

        self.mb_loader.set_rollout_state(self._initial_state, update_dones)

    def _rollout_step(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects actions for a single environment step during rollouts,
        advancing the recurrent state.

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
            mean, log_std, values, self._next_state = self.agent(
                obs,
                self._next_state,
                self._next_done,
            )
            actions, log_probs = self.sampler.sample(mean, log_std)

        return actions, log_probs, values.flatten()

    def _bootstrap_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes the value estimate for the observation following the
        rollout's final step, used to bootstrap the GAE. Does not
        advance the recurrent state.

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
            return self.agent(obs, self._next_state, self._next_done)[2]

    def _evaluate(  # type: ignore[override]
        self,
        minibatch: LSTMMiniBatchData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluates a mini-batch of rollout actions under the current
        policy, replaying trajectories from the recurrent state that
        entered the rollout.

        Parameters
        ----------
        minibatch : LSTMMiniBatchData
            A mini-batch of whole-trajectory experience

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
        mean, log_std, new_values, _ = self.agent(
            minibatch.obs,
            minibatch.initial_state,
            minibatch.dones,
        )
        log_probs, entropy = self.sampler.evaluate(
            mean,
            log_std,
            actions=minibatch.actions,
        )
        return log_probs, entropy, new_values
