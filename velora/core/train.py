# Copyright 2025 Achronus
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

from typing import List

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax

from velora.config.settings import AgentTrainerSettings, RuleTrainerSettings
from velora.config.state import AgentTrainerState, BundledHiddenStates, CollectState
from velora.core.agent import DiscoAgent, PolicyAgent
from velora.core.buffer import MixedBuffer
from velora.gym.utils import make_atari_env


class RuleTrainer:
    """
    Discovers a Reinforcement Learning (RL) update rule by meta-training across environments.

    `RuleTrainer` handles the outer loop of target rule learning (DiscoRL): managing a population of
    `AgentTrainer`s across diverse environments, computing meta-gradients from
    their learning progress, and updating the target agent to improve the
    collective performance of all agents.

    The discovered rule is encoded in the target agent's parameters. Once trained,
    these parameters can be frozen and used with `AgentTrainer` alone to train
    new agents in unseen environments.

    Parameters
    ----------
    envs : List[str]
        Environments to use for rule discovery
    config : RuleTrainerSettings
        Configuration for meta-training
    seed : int (optional)
        Random number generator seed. Default is `42`
    """

    def __init__(
        self,
        envs: List[str],
        config: RuleTrainerSettings,
        seed: int = 42,
    ) -> None:
        self.envs = envs
        self.config = config
        self.key = jax.random.key(seed)

        self.rule_key, meta_key, trainer_key = jax.random.split(self.key, 3)
        trainer_keys = jax.random.split(trainer_key, len(envs))

        self.meta_agent = DiscoAgent(config=config.disco_agent, key=meta_key)
        self.meta_optim = optax.chain(
            optax.clip_by_global_norm(config.meta_grad_clip),
            optax.adam(config.meta_lr),
        )
        self.meta_optim_state = self.meta_optim.init(self.meta_agent.params)

        self.trainers = [
            AgentTrainer(
                env_name,
                self.config.agent_trainer_config(),
                key=trainer_keys[i],
            )
            for i, env_name in enumerate(envs)
        ]

        self.num_envs = len(self.envs)
        self.current_env_idx = 0

    def train(self) -> None:
        pass


class AgentTrainer:
    """
    Trains a single `PolicyAgent` in a single environment using a learned update rule.

    `AgentTrainer` handles the inner loop of target rule learning (DiscoRL): collecting trajectories,
    sampling from the buffer, generating targets via the target agent, and
    updating the `PolicyAgent` to minimize prediction error against those targets.

    This class is designed to be instantiated multiple times by `RuleTrainer`,
    with each instance training an agent in a different environment. During
    meta-training, it exposes differentiable update steps that allow gradients
    to flow back through the agent's learning process.

    Parameters
    ----------
    env_name : str
        The gymnasium environment name to train on
    config : AgentTrainerSettings
        Configuration for individual agent training
    key : chex.PRNGKey
        Random number generator key
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    """

    def __init__(
        self,
        env_name: str,
        config: AgentTrainerSettings,
        *,
        key: chex.PRNGKey,
        jit_compile: bool = False,
    ) -> None:
        self.config = config
        self.env_name = env_name
        self.envs = make_atari_env(self.env_name, self.config.num_envs)
        self.jit_compile = jit_compile

        action_space: gym.spaces.Discrete = self.envs.single_action_space  # type: ignore
        obs_space: gym.spaces.Box = self.envs.single_observation_space  # type: ignore

        self.n_actions = action_space.n.item()
        self.key, agent_key, target_key, buffer_key = jax.random.split(key, 4)

        self.policy_agent = PolicyAgent(
            obs_space,
            action_space,
            config=self.config.agent,
            key=agent_key,
            jit_compile=self.jit_compile,
        )

        self.target_agent = PolicyAgent(
            obs_space,
            action_space,
            config=self.config.agent,
            key=target_key,
            jit_compile=self.jit_compile,
        )

        self.buffer = MixedBuffer(buffer_key, config=self.config.buffer)

        # Init agent state
        current_obs, _ = self.envs.reset()

        self.state = AgentTrainerState.create(
            self.policy_agent.get_params(),
            self.target_agent.get_params(),
            self._create_optim(),
            current_obs=current_obs,
        )

        # Checkpointing
        self.cp_manager = CheckpointManager(
            self.env_name, config=self.config.checkpoint
        )

        # Warm policy networks and buffer
        self._warm(obs_space)

    def _create_optim(self) -> optax.GradientTransformation:
        """
        Create optimizer chain for agent and value network training.

        Uses Adan optimizer (without denominator) with gradient clipping
        and scaled learning rate.

        Returns
        -------
        optim : optax.GradientTransformation
            Configured optimizer chain
        """
        return optax.chain(
            scale_by_adan_no_denom(),
            optax.clip(self.config.agent.max_grad_norm),
            optax.scale(-self.config.agent.lr),
        )

    def _warm(self, obs_space: gym.spaces.Box) -> None:
        """
        Performs an initial forward pass through the agent networks and buffer to:
            1. Materialize parameters before optimizer init
            2. JIT compile

        Parameters
        ----------
        obs_space : gym.spaces.Box
            The environment observation space
        """
        dummy_obs = jnp.zeros(
            (self.config.num_envs, *obs_space.shape),
            dtype=obs_space.dtype,
        )
        _ = self.policy_agent(dummy_obs)
        _ = self.target_agent(dummy_obs)

        # Warm buffer
        self.collect()
        self.buffer.clear()

    def _compute_policy_loss(
        self,
        batch: BufferSamples,
        targets: DiscoAgentOutput,
    ) -> AgentLosses:
        """
        Computes the policy agent losses for a single timestep.

        The loss consists of:
        - KL divergence between target policy (`π̂ `) and agent policy (`π`)
        - KL divergence between target y (`ŷ`) and agent `y`
        - KL divergence between target z (`ẑ`) and agent `z` (for action taken)
        - Auxiliary policy prediction loss

        Parameters
        ----------
        batch : BufferSamples
            Batch of experience from the replay buffer
        targets : DiscoAgentOutput
            Targets generated by the meta-network `(π̂, ŷ, ẑ)`

        Returns
        -------
        losses : AgentLosses
            An object containing the agent losses
        """
        preds = batch.preds

        pi_loss = compute_kl_loss(targets.pi, preds.pi).mean()
        y_loss = compute_kl_loss(targets.y, preds.y).mean()

        # z_loss - gather by action taken
        action_preds = jnp.take_along_axis(
            preds.z,
            batch.actions[..., None, None],  # (B, T, 1, 1), # type: ignore
            axis=2,
        ).squeeze(2)  # (B, T, Z)
        z_loss = compute_kl_loss(targets.z, action_preds).mean()

        # aux_pi_loss - predict next timesteps policy
        aux_pi_loss = compute_aux_policy_loss(
            preds.aux_pi[:, :-1],  # type: ignore
            preds.pi[:, 1:],  # type: ignore
            batch.discounts[:, :-1],  # type: ignore
        ).mean()

        return AgentLosses(
            pi=pi_loss,
            y=y_loss,
            z=z_loss,
            aux_pi=aux_pi_loss,
        ).compute_total(self.config.loss_costs)

    def _update_policy_step(
        self,
        batch: BufferSamples,
        targets: DiscoAgentOutput,
    ) -> Tuple[AgentLosses, Dict[str, Any]]:
        """
        Performs a single policy update step.

        Parameters
        ----------
        batch : BufferSamples
            Batch of experience from the buffer
        targets : DiscoAgentOutput
            Targets generated by the meta-network `(π̂, ŷ, ẑ)`

        Returns
        -------
        losses : AgentLosses
            An object containing the agent losses
        metrics : Dict[str, Any]
            Training metrics
        """
        params = self.state.params.policy

        def loss_fn(p):
            self.policy_agent.update_params(p)
            return self._compute_policy_loss(batch, targets)

        losses, grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        updates, new_optim_state = self.state.optim.update(
            grads, self.state.opt_state, params
        )
        new_params = optax.apply_updates(params, updates)

        # Update state
        self.state.update_policy_params(
            new_params,  # type: ignore
            new_optim_state,
            self.config.tau,
        )

        # Sync params
        self.policy_agent.update_params(new_params)  # type: ignore
        self.target_agent.update_params(self.state.params.target)

        metrics = {
            "grad_norm": optax.global_norm(grads),
            "steps_trained": self.state.steps_trained,
        }

        return losses, metrics

    def step(self, targets: DiscoAgentOutput) -> Tuple[AgentLosses, Dict[str, Any]]:
        """
        Perform a single training step.

        Parameters
        ----------
        targets : DiscoAgentOutput
            Targets from the meta-network

        Returns
        -------
        losses : AgentLosses
            Policy loss components
        metrics : Dict[str, Any]
            Training metrics
        """
        batch = self.buffer.sample(self.config.batch_size)
        return self._update_policy_step(batch, targets)

    def collect(self) -> CollectState:
        """
        Collects a trajectory and adds it to the buffer.

        Returns
        -------
        state : CollectState
            Trajectory collection state
        """
        # Sync parameters
        self.policy_agent.update_params(self.state.params.policy)
        self.target_agent.update_params(self.state.params.target)

        # Tracking
        collect_state = CollectState()
        current_obs = self.state.current_obs
        hidden = self.state.hidden

        # Trajectory collection
        for _ in range(self.buffer.seq_len):
            preds, h_state = self.policy_agent(
                current_obs,
                ocm_h_state=hidden.ocm,
                acm_h_state=hidden.acm,
            )
            target_preds, target_h_state = self.target_agent(
                current_obs,
                ocm_h_state=hidden.target_ocm,
                acm_h_state=hidden.target_acm,
            )

            # Env step
            actions = self.policy_agent.act(preds.pi)
            next_obs, rewards, discounts, info = self.envs.step(actions)

            # Accumulate step data and episode stats
            collect_state = collect_state.append_step(
                actions, rewards, discounts, preds, target_preds
            )
            collect_state = collect_state.record_episodes(info)

            # Update and reset hidden states on episode boundaries
            hidden.update(h_state, target_h_state)
            hidden.reset_on_done(discounts, self.n_actions)

            current_obs = next_obs

        # Update state with new env state
        self.state.update_hidden_states(hidden)
        self.state.update_obs(current_obs)

        # Add to buffer
        batched_samples = collect_state.to_batches()
        self.buffer.add(**batched_samples.to_dict())

        return collect_state
