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

from typing import Dict, List, Tuple

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from velora.base.rollouts import Rollout, RolloutStack
from velora.core.outputs import BufferSamples
from velora.disco.agent import DiscoAgent, DiscoValueAgent, PolicyAgent
from velora.disco.ema import EMAState, MovingAverage
from velora.disco.outputs import AgentLosses, DiscoAgentOutput, ValueOutputs
from velora.disco.settings import AgentTrainerSettings, RuleTrainerSettings
from velora.disco.state import AgentTrainerState, CollectState, RuleTrainerState
from velora.disco.utils.compute import compute_value_outputs
from velora.disco.utils.loss import (
    compute_entropy_loss,
    compute_meta_reg_loss,
    compute_policy_gradient_loss,
    compute_policy_loss,
)
from velora.gym.utils import make_atari_env
from velora.nn.optim import scale_by_adan_no_denom
from velora.tracking.logger import MetricsLogger
from velora.tracking.manager import CheckpointManager


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
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    """

    def __init__(
        self,
        envs: List[str],
        *,
        config: RuleTrainerSettings,
        seed: int = 42,
        jit_compile: bool = False,
    ) -> None:
        self.env_names = envs
        self.num_envs = len(envs)
        self.config = config
        self.jit_compile = jit_compile

        self.current_env_idx = 0
        self.meta_step = 0

        # Init logger
        self.logger = MetricsLogger(self.config.logger)
        self.logger.add_writer("meta")

        for name in self.env_names:
            self.logger.add_writer(f"envs/{name}")

        # Configure RNG keys
        key = jax.random.key(seed)
        self.key, meta_key, *trainer_keys = jax.random.split(key, self.num_envs + 2)

        # Init meta-agent and optimizer
        self.meta_agent = DiscoAgent(
            config=config.disco_agent,
            key=meta_key,
            jit_compile=self.jit_compile,
        )
        self.meta_optim = optax.chain(
            optax.clip_by_global_norm(config.meta_grad_clip),
            optax.adam(config.meta_lr),
        )

        # Init state
        self.state = RuleTrainerState.create(
            self.meta_optim.init(self.meta_agent.get_params()),
            self.num_envs,
        )

        # Init agent trainers
        self.trainers = [
            AgentTrainer(
                env_name,
                self.config.agent_trainer_config(),
                key=trainer_keys[i],
                logger=self.logger,
                writer_name=f"envs/{env_name}",
                jit_compile=self.jit_compile,
            )
            for i, env_name in enumerate(self.env_names)
        ]

        # Checkpointing
        self.cp_manager = CheckpointManager(
            "rule_trainer",
            config=self.config.checkpoint,
        )

    def train(self) -> None:
        """
        Performs meta-training loop to discover an RL update rule.

        Includes -
        1. For each meta-step:

            a. Collect trajectories from all environments
            b. For each agent, perform inner loop updates
            c. Compute meta-gradients through the learning process
            d. Average gradients across agents and update meta-network

        2. Log metrics and checkpoints periodically
        """
        for step in range(self.config.n_steps):
            self.meta_step = step

            meta_grads_list = []
            for idx, trainer in enumerate(self.trainers):
                trainer.collect()

                # Compute meta_gradient for this agent
                meta_grad = self._compute_meta_gradient(idx)
                meta_grads_list.append(meta_grad)

            # Average meta-gradients across all agents
            avg_meta_grad = jax.tree.map(
                lambda *grads: jnp.mean(jnp.stack(grads), axis=0),
                *meta_grads_list,
            )
            self._apply_meta_update(avg_meta_grad)

            # Log metrics
            self.logger.log(
                "meta",
                self.meta_step,
                {
                    "meta/grad_norm": float(optax.global_norm(avg_meta_grad)),
                    "meta/step": self.meta_step,
                },
            )

            # Checkpoint periodically
            if step % self.config.checkpoint.freq == 0:
                self.save_checkpoint()

        # Final cleanup
        self.close()

    def _compute_meta_gradient(self, trainer_idx: int) -> chex.ArrayTree:
        """
        Compute meta-gradient for a single agent through the inner learning loop.

        Uses NNX split/merge pattern for pure functional tracing compatibility
        with JAX transformations (jit, grad).

        Parameters
        ----------
        trainer_idx : int
            Index of the agent trainer to update

        Returns
        -------
        meta_grad : chex.ArrayTree
            Gradients w.r.t. meta-network parameters
        """
        trainer = self.trainers[trainer_idx]

        # Get current state
        policy_params = trainer.state.params.policy
        opt_state = trainer.state.opt_state
        disco_h, meta_h = self.h_state.get(trainer_idx)

        # Split meta-agent into graphdef and state for pure functional tracing
        # Note: separates the module structure from its parameters
        meta_graphdef, meta_state = nnx.split(self.meta_agent)

        # Pre-sample and stack batches for scan compatibility
        train_batches = [
            trainer.buffer.sample(self.config.batch_size)
            for _ in range(self.config.n_updates)
        ]
        stacked_batches = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0),
            *train_batches,
        )
        value_batch = trainer.buffer.sample(self.config.batch_size)

        # Get config values
        entropy_coef = self.config.entropy_coef
        reg_scale = self.config.reg_scale
        kl_reg = self.config.kl_reg
        loss_costs = trainer.config.loss_costs

        def meta_loss_fn(meta_state: nnx.State) -> Tuple[chex.Array, MetaGradientAux]:
            """
            Pure meta-loss computation with no side effects.

            All state is passed explicitly via arguments and returned via outputs.
            """
            # Reconstruct meta-agent from state
            disco_agent: DiscoAgent = nnx.merge(meta_graphdef, meta_state)

            def inner_step(carry, batch: BufferSamples):
                """Single inner loop update step."""
                p_params, d_h, m_h = carry

                # Generate targets from disco agent
                targets, d_h, m_h = disco_agent(batch, d_h, m_h)

                # Compute policy gradients
                # Note: Gradient w.r.t p_params will be zero since batch.preds
                # doesn't depend on p_params. This simulates the learning process.
                def loss_fn(params):
                    return compute_policy_loss(
                        targets,
                        batch.preds.pi,
                        batch.preds.y,
                        batch.preds.z,
                        batch.preds.aux_pi,
                        batch.actions,
                        batch.discounts,
                        loss_costs,
                    )

                (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(p_params)

                # Apply inner update
                updates, _ = trainer.state.optim.update(grads, opt_state, p_params)
                p_params = optax.apply_updates(p_params, updates)

                return (p_params, d_h, m_h), targets

            # Run inner loop
            init_carry = (policy_params, disco_h, meta_h)
            (final_p_params, final_d_h, final_m_h), all_targets = jax.lax.scan(
                inner_step, init_carry, stacked_batches
            )

            # Get last targets for regularization loss
            last_targets = jax.tree.map(lambda x: x[-1], all_targets)

            # Compute meta-loss using value batch
            value_outs = compute_value_outputs(
                value_batch,
                trainer.state.value.adv_ema,
                trainer.state.value.td_ema,
                trainer.config.value.gamma,
                trainer.config.value.td_lambda,
                update_ema=False,  # Pure - no side effects
            )

            pg_loss = compute_policy_gradient_loss(
                value_batch.preds.pi,
                value_batch.actions,
                value_outs.normalized_advantages,
            )
            entropy_loss = compute_entropy_loss(
                value_batch.preds.pi,
                entropy_coef,
            )
            reg_loss = compute_meta_reg_loss(
                last_targets, value_batch.target_preds.pi, reg_scale, kl_reg
            )
            meta_loss = pg_loss + entropy_loss + reg_loss

            aux = MetaGradientAux(
                disco_h=final_d_h,
                meta_h=final_m_h,
                pg_loss=pg_loss,
                entropy_loss=entropy_loss,
                reg_loss=reg_loss,
                value_outs=value_outs,
            )
            return meta_loss, aux

        # Compute gradients w.r.t meta-network state (parameters)
        (meta_loss, aux), meta_grad = jax.value_and_grad(meta_loss_fn, has_aux=True)(
            meta_state
        )
        aux: MetaGradientAux = aux

        # Update hidden state (outside traced function)
        self.h_state = self.h_state.update(trainer_idx, aux.disco_h, aux.meta_h)

        # Log metrics
        self._log_meta_metrics(
            trainer_idx,
            aux.pg_loss,
            aux.entropy_loss,
            aux.reg_loss,
            meta_loss,
            aux.value_outs,
        )

        # Update value function (outside traced function)
        trainer.update_value(value_batch)

        return meta_grad

    def _log_meta_metrics(
        self,
        trainer_idx: int,
        pg_loss: chex.Array,
        entropy_loss: chex.Array,
        reg_loss: chex.Array,
        total_loss: chex.Array,
        value_outs: ValueOutputs,
    ) -> None:
        """
        Log meta-training metrics for a single environment.

        Parameters
        ----------
        trainer_idx : int
            Index of the current agent trainer
        pg_loss : chex.Array
            Policy gradient loss
        entropy_loss : chex.Array
            Entropy regularization loss
        reg_loss : chex.Array
            L2 and KL regularization loss on meta-network targets
        total_loss : chex.Array
            Sum of all loss components
        value_outs : ValueOutputs
            Value function outputs containing advantage estimates
        """
        metrics = {
            "meta/pg_loss": float(pg_loss),
            "meta/entropy_loss": float(entropy_loss),
            "meta/reg_loss": float(reg_loss),
            "meta/total_loss": float(total_loss),
            "meta/advantages": float(jnp.mean(value_outs.advantages)),
            "meta/normalized_advantages": float(
                jnp.mean(value_outs.normalized_advantages)
            ),
        }
        self.logger.log(f"envs/{self.env_names[trainer_idx]}", self.meta_step, metrics)

    def _apply_meta_update(self, meta_grad: chex.ArrayTree) -> None:
        """
        Apply meta-gradient update to meta-network parameters.

        Parameters
        ----------
        meta_grad : chex.ArrayTree
            Averaged gradients across all agents
        """
        # Split to get current state in same structure as gradients
        _, meta_state = nnx.split(self.meta_agent)

        updates, self.meta_optim_state = self.meta_optim.update(
            meta_grad,
            self.meta_optim_state,
            meta_state,  # type: ignore
        )
        new_state = optax.apply_updates(meta_state, updates)  # type: ignore

        # Merge updated state back into agent
        nnx.update(self.meta_agent, new_state)

    def save_checkpoint(self, force: bool = False) -> bool:
        """
        Save current state to checkpoint.

        Saves both meta-network state and all individual trainer states.

        Parameters
        ----------
        force : bool (optional)
            Force save even if within save interval. Default is `False`

        Returns
        -------
        saved : bool
            Whether checkpoint was actually saved
        """
        if not force and not self.cp_manager.should_save(self.meta_step):
            return False

        state = RuleTrainerState(
            meta_step=self.meta_step,
            meta_params=self.meta_agent.get_params(),
            meta_optim_state=self.meta_optim_state,
            hidden_states=self.hidden_states,
        )

        self.cp_manager.save(self.meta_step, state, force=True)

        # Save individual trainer states
        for trainer in self.trainers:
            trainer.save_checkpoint(force=True)

        return True

    def load_checkpoint(self, step: int | None = None) -> bool:
        """
        Restore state from a checkpoint.

        Restores both meta-network state and all individual trainer states.

        Parameters
        ----------
        step : int (optional)
            Specific step to restore, or `None` for latest. Default is `None`

        Returns
        -------
        success : bool
            Whether restoration was successful
        """
        # Create abstract state for safe restoration
        abstract_state = RuleTrainerState(
            meta_step=0,
            meta_params=jax.tree.map(
                ocp.utils.to_shape_dtype_struct,
                self.meta_agent.get_params(),
            ),
            meta_optim_state=jax.tree.map(
                ocp.utils.to_shape_dtype_struct,
                self.meta_optim_state,
            ),
            hidden_states=jax.tree.map(
                lambda x: ocp.utils.to_shape_dtype_struct(x) if x is not None else None,
                self.hidden_states,
            ),
        )

        restored_state: RuleTrainerState = self.cp_manager.restore(step, abstract_state)  # type: ignore

        if restored_state is None:
            return False

        # Restore meta state
        self.meta_step = restored_state.meta_step
        self.meta_agent.update_params(restored_state.meta_params)
        self.meta_optim_state = restored_state.meta_optim_state
        self.hidden_states = restored_state.hidden_states

        # Restore individual trainer states
        for trainer in self.trainers:
            trainer.load_checkpoint(step)

        return True

    def reset_trainer(self, trainer_idx: int) -> None:
        """
        Reset a trainer to initial state.

        Used when an agent has consumed its experience budget during
        meta-training. Creates a fresh agent while preserving the
        meta-network parameters.

        Parameters
        ----------
        trainer_idx : int
            Index of the trainer to reset
        """
        env_name = self.env_names[trainer_idx]

        # Close existing trainer
        self.trainers[trainer_idx].close()

        # Fresh RNG keys
        self.rule_key, new_key = jax.random.split(self.rule_key)

        # Init new trainer
        self.trainers[trainer_idx] = AgentTrainer(
            env_name,
            self.config.agent_trainer_config(),
            key=new_key,
            logger=self.logger,
            writer_name=f"envs/{env_name}",
            jit_compile=self.jit_compile,
        )
        self.h_state.reset(trainer_idx)

    def close(self) -> None:
        """Clean up resources."""
        self.cp_manager.close()

        for trainer in self.trainers:
            trainer.close()


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
    logger : MetricsLogger
        The Tensorboard metrics logger for metric tracking
    writer_name : str
        The name of the metrics logger writer to use
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    """

    def __init__(
        self,
        env_name: str,
        config: AgentTrainerSettings,
        *,
        key: chex.PRNGKey,
        logger: MetricsLogger,
        writer_name: str,
        jit_compile: bool = False,
    ) -> None:
        self.config = config
        self.env_name = env_name

        self.envs = make_atari_env(self.env_name, self.config.num_vec_envs)

        self.logger = logger
        self.writer_name = writer_name
        self.jit_compile = jit_compile

        action_space: gym.spaces.Discrete = self.envs.single_action_space  # type: ignore
        obs_space: gym.spaces.Box = self.envs.single_observation_space  # type: ignore

        self.n_actions = action_space.n.item()
        self.key, agent_key, target_key, value_key = jax.random.split(key, 4)

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

        self.value_agent = DiscoValueAgent(
            obs_space,
            self.config.agent.n_hidden,
            config=self.config.value,
            key=value_key,
            sparsity=self.config.agent.sparsity,
            jit_compile=self.jit_compile,
        )

        # Init state
        current_obs, _ = self.envs.reset()

        self.policy_optim = self._create_optim()
        self.value_optim = self._create_optim()

        self.ema_utils = MovingAverage(self.config.ema)
        self.state = AgentTrainerState.create(
            self.policy_optim.init(self.policy_agent.get_params()),
            self.value_optim.init(self.value_agent.get_params()),
            current_obs=current_obs,
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
        Performs an initial forward pass through the agent networks:
            1. Materialize parameters before optimizer init
            2. JIT compile

        Parameters
        ----------
        obs_space : gym.spaces.Box
            The environment observation space
        """
        dummy_obs = jnp.zeros(
            (self.config.num_vec_envs, *obs_space.shape),
            dtype=obs_space.dtype,
        )
        _ = self.policy_agent(dummy_obs)
        _ = self.target_agent(dummy_obs)
        _ = self.value_agent(dummy_obs)

    def log(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics to `MetricsLogger` if logger is available.

        Parameters
        ----------
        metrics : Dict[str, float]
            Mapping of metric names to scalar values
        """
        if self.logger:
            self.logger.log(self.writer_name, self.state.steps_trained, metrics)

    def update_policy(
        self,
        rollout: Rollout,
        targets: DiscoAgentOutput,
    ) -> None:
        """
        Update policy network parameters.

        Parameters
        ----------
        rollout : Rollout
            A trajectory of experience
        targets : DiscoAgentOutput
            Targets generated by the meta-network `(π̂, ŷ, ẑ)`
        """
        policy_params = self.policy_agent.get_params()

        # Compute policy gradients
        def loss_fn(params) -> Tuple[chex.Array, AgentLosses]:
            """Compute policy loss."""
            return compute_policy_loss(
                targets,
                rollout.preds.pi,
                rollout.preds.y,
                rollout.preds.z,
                rollout.preds.aux_pi,
                rollout.actions,
                rollout.discounts,
                self.config.loss_costs,
            )

        (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_params)

        # Update parameters
        updates, new_opt_state = self.policy_optim.update(
            grads,
            self.state.policy_opt_state,
            policy_params,
        )
        new_params = optax.apply_updates(policy_params, updates)

        self.policy_agent.update_params(new_params)  # type: ignore
        self.target_agent.soft_param_update(self.config.tau, new_params)  # type: ignore

        # Update state
        self.state = self.state.update_policy_opt(new_opt_state)

        # Log metrics
        losses: AgentLosses = losses
        metrics = {
            **losses.to_metrics(),
            "inner/grad_norm": optax.global_norm(grads),
            "inner/steps_trained": self.state.steps_trained,
        }
        self.log(metrics)

    def update_value(self, rollout: Rollout) -> ValueOutputs:
        """
        Update value network parameters.

        Parameters
        ----------
        rollout : Rollout
            A trajectory of experience

        Returns
        -------
        values_outs : ValueOutputs
            Value function outputs
        """
        value_params = self.value_agent.get_params()

        def loss_fn(params) -> Tuple[chex.Array, ValueOutputs, EMAState, EMAState]:
            """Compute value loss."""
            value_outs, adv_ema, td_ema = compute_value_outputs(
                rollout,
                self.ema_utils,
                self.state.adv_ema,
                self.state.td_ema,
                self.config.value.gamma,
                self.config.value.td_lambda,
            )
            net_out = value_outs.value[:-1]  # type: ignore

            # Value loss from normalized TD
            # loss = 0.5 * (value - stop_grad[value + TD])^2
            value_target = jax.lax.stop_gradient(net_out + value_outs.normalized_td)
            value_loss = 0.5 * jnp.square(net_out - value_target).mean()
            value_loss = self.config.loss_costs.value * value_loss

            return value_loss, value_outs, adv_ema, td_ema

        # Compute gradients
        (v_loss, v_outs, adv_ema, td_ema), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(value_params)

        # Apply optimizer
        updates, new_opt_state = self.value_optim.update(
            grads,
            self.state.value_opt_state,
            value_params,
        )
        new_params = optax.apply_updates(value_params, updates)

        # Update state
        self.value_agent.update_params(new_params)  # type: ignore
        self.state = self.state.update_value_opt(new_opt_state)
        self.state = self.state.update_ema(adv_ema, td_ema)

        # Log metrics
        v_outs: ValueOutputs = v_outs
        metrics = {
            "value/loss": v_loss,
            **v_outs.to_metrics(),
            "value/grad_norm": optax.global_norm(grads),
        }
        self.log(metrics)

        return v_outs

    def collect(self) -> Rollout:
        """
        Collects a trajectory by interacting with the environment.

        Returns
        -------
        rollout : Rollout
            A single trajectory
        """
        obs = self.state.current_obs
        hidden = self.state.hidden
        collect_state = CollectState()

        # Trajectory collection
        for _ in range(self.config.seq_len):
            preds, h_policy = self.policy_agent(
                obs,
                ocm_h_state=hidden.policy_ocm,
                acm_h_state=hidden.policy_acm,
            )
            target_preds, h_target = self.target_agent(
                obs,
                ocm_h_state=hidden.target_ocm,
                acm_h_state=hidden.target_acm,
            )

            values, h_value = self.value_agent(
                obs,
                h_state=hidden.value,
            )

            # Env step
            actions = self.policy_agent.act(preds.pi)
            next_obs, rewards, discounts, info = self.envs.step(actions)

            # Store step data and episode stats
            collect_state = collect_state.append_step(
                actions, rewards, discounts, values, preds, target_preds
            )
            collect_state = collect_state.record_episodes(info)

            # Update and reset hidden states on episode boundaries
            hidden = hidden.update(h_policy, h_target, h_value)
            hidden = hidden.reset_on_done(discounts, self.n_actions)

            obs = next_obs

        # Update trainer state
        self.state = self.state.update_hidden(hidden)
        self.state = self.state.update_obs(obs)

        # Log episodic metrics
        if metrics := collect_state.episode_metrics():
            self.log(metrics)

        return collect_state.to_rollout()

    def close(self) -> None:
        """Clean up resources."""
        self.envs.close()
