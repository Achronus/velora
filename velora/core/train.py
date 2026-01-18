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
        self.envs = make_atari_env(env_name, self.config.num_envs)
        self.key = key
        self.jit_compile = jit_compile

        action_space: gym.spaces.Discrete = self.envs.single_action_space  # type: ignore
        obs_space: gym.spaces.Box = self.envs.single_observation_space  # type: ignore

        agent_key, target_key, buffer_key = jax.random.split(self.key, 3)
        self.state: AgentTrainerState | None = None

        self.agent = PolicyAgent(
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
        self.buffer = MixedBuffer(buffer_key, config=config.buffer)

    def collect(self) -> None:
        """
        Collects a trajectory and adds it to the buffer.
        """
        obs, _ = self.envs.reset()

        hidden = BundledHiddenStates()
        state = CollectState()

        # Trajectory collection
        for _ in range(self.buffer.seq_len):
            preds, h_state = self.agent(
                obs,
                ocm_h_state=hidden.ocm,
                acm_h_state=hidden.acm,
            )
            target_preds, target_h_state = self.target_agent(
                obs,
                ocm_h_state=hidden.target_ocm,
                acm_h_state=hidden.target_acm,
            )

            # Env step
            actions = self.agent.act(preds.pi)
            next_obs, rewards, dones, info = self.envs.step(actions)

            # Accumulate step data and episode stats
            state = state.append_step(actions, rewards, dones, preds, target_preds)
            state = state.record_episodes(info)

            # Update and reset hidden states on episode boundaries
            hidden = hidden.update(h_state, target_h_state)
            hidden = hidden.reset_on_done(dones, self.agent.n_actions)

            obs = next_obs

        # Convert to batched arrays and add to buffer
        batched_samples = state.to_batches()
        self.buffer.add(**batched_samples.to_dict())
