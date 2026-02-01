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

from velora.config.state import (
    AgentTrainerState,
    CollectState,
    RuleTrainerHiddenStates,
    RuleTrainerState,
)
from velora.core.buffer import MixedBuffer
from velora.core.outputs import BufferSamples
from velora.disco.agent import DiscoAgent, DiscoValueAgent, PolicyAgent
from velora.disco.outputs import (
    AgentLosses,
    DiscoAgentOutput,
    MetaGradientAux,
    ValueOutputs,
)
from velora.disco.settings import AgentTrainerSettings, RuleTrainerSettings
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
        self.num_envs = len(self.env_names)
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
        self.key = jax.random.key(seed)
        self.rule_key, meta_key, trainer_key = jax.random.split(self.key, 3)
        trainer_keys = jax.random.split(trainer_key, self.num_envs)

        # Init agents
        self.meta_agent = DiscoAgent(
            config=config.disco_agent,
            key=meta_key,
            jit_compile=self.jit_compile,
        )
        self.meta_optim = optax.chain(
            optax.clip_by_global_norm(config.meta_grad_clip),
            optax.adam(config.meta_lr),
        )
        self.meta_optim_state = self.meta_optim.init(self.meta_agent.get_params())

        self.h_state = RuleTrainerHiddenStates.create(self.num_envs)

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
        n_updates = self.config.n_updates

        def meta_loss_fn(meta_params) -> chex.Array:
            # Set current params
            self.meta_agent.update_params(meta_params)

            disco_h, meta_h = self.h_state.get(trainer_idx)
            policy_params = trainer.state.params.policy

            # Inner loop: perform n updates
            last_targets = None
            for _ in range(n_updates):
                batch = trainer.buffer.sample(self.config.batch_size)

                targets, disco_h, meta_h = self.meta_agent(
                    batch,
                    disco_h_state=disco_h,
                    meta_h_state=meta_h,
                )
                last_targets = targets

                def policy_loss_fn(params):
                    trainer.policy_agent.update_params(params)
                    return trainer._compute_policy_loss(batch, targets)

                _, grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(
                    policy_params
                )

                # Apply inner update
                updates, _ = trainer.state.optim.update(
                    grads,
                    trainer.state.opt_state,
                    policy_params,
                )
                policy_params = optax.apply_updates(policy_params, updates)

            # Update hidden states and params
            self.h_state.update(trainer_idx, disco_h, meta_h)
            trainer.policy_agent.update_params(policy_params)  # type: ignore

            # Compute values function outputs
            value_batch = trainer.buffer.sample(self.config.batch_size)
            value_outs = trainer.get_values(value_batch)

            # Compute meta-loss
            pg_loss = self._policy_gradient_loss(
                value_batch.preds.pi,
                value_batch.actions,
                value_outs.normalized_advantages,
            )
            entropy_loss = compute_entropy_loss(
                value_batch.preds.pi,
                self.config.entropy_coef,
            )
            reg_loss = self._compute_meta_reg_loss(last_targets, value_batch)  # type: ignore
            meta_loss = pg_loss + entropy_loss + reg_loss

            # Log metrics
            self._log_meta_metrics(
                trainer_idx,
                pg_loss,
                entropy_loss,
                reg_loss,
                meta_loss,
                value_outs,
            )

            return meta_loss

        # Compute gradients w.r.t meta-network params
        meta_params = self.meta_agent.get_params()
        _, meta_grad = jax.value_and_grad(meta_loss_fn, has_aux=True)(meta_params)

        # Update value function
        batch = trainer.buffer.sample(self.config.batch_size)
        trainer.update_value_params(batch)

        return meta_grad

    def _log_meta_metrics(
        self,
        trainer_idx: int,
        pg_loss: chex.Array,
        entropy_loss: chex.Array,
        reg_loss: chex.Array,
        total_loss: chex.Array,
        value_outs: DiscoValueOutputs,
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
        value_outs : DiscoValueOutputs
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

    def _policy_gradient_loss(
        self, logits: chex.Array, actions: chex.Array, advantages: chex.Array
    ) -> chex.Array:
        """
        Compute differentiable policy gradient loss (REINFORCE-style).

        Calculates `-log π(a|s) * A(s, a)` for each timestep. Advantages are
        stopped from gradient flow as they should not influence the meta-network
        through this path.

        Parameters
        ----------
        logits : chex.Array
            Policy logits. Shape: `(B, T, A)`
        actions : chex.Array
            Actions taken. Shape: `(B, T, 1)`
        advantages : chex.Array
            Advantage estimates. Shape: `(T, B)`

        Returns
        -------
        loss : chex.Array
            Per-timestep policy gradient loss. Shape: `(B, T)`
        """
        advantages = jnp.transpose(advantages)  # (B, T)
        actions = jnp.squeeze(actions, axis=-1)  # (B, T)

        log_pi = jax.nn.log_softmax(logits)
        log_pi_a = rlax.batched_index(log_pi, actions)
        return -log_pi_a * jax.lax.stop_gradient(advantages)  # type: ignore

    def _compute_meta_reg_loss(
        self,
        targets: DiscoAgentOutput,
        batch: BufferSamples,
    ) -> chex.Array:
        """
        Compute regularization terms for meta-network outputs.

        Includes -
        - L2 regularization on target means (prevent divergence)
        - KL divergence between targets and current policy (stability)

        Parameters
        ----------
        targets : DiscoAgentOutput
            Generated targets `(π̂, ŷ, ẑ)`
        batch : BufferSamples
            Batch of experience

        Returns
        -------
        reg_loss : chex.Array
            Total regularization loss
        """
        pi_reg = compute_l2_mean_penalty(targets.pi)
        y_reg = compute_l2_mean_penalty(targets.y)
        z_reg = compute_l2_mean_penalty(targets.z)

        target_kl = compute_kl_loss(
            jax.lax.stop_gradient(batch.target_preds.pi),
            targets.pi,
        ).mean()

        l2_reg = self.config.reg_scale * (pi_reg + y_reg + z_reg)
        kl_reg = self.config.kl_reg * target_kl
        return l2_reg + kl_reg

    def _apply_meta_update(self, meta_grad: chex.ArrayTree) -> None:
        """
        Apply meta-gradient update to meta-network parameters.

        Parameters
        ----------
        meta_grad : chex.ArrayTree
            Averaged gradients across all agents
        """
        meta_params = self.meta_agent.get_params()

        updates, self.meta_optim_state = self.meta_optim.update(
            meta_grad,
            self.meta_optim_state,
            meta_params,
        )
        new_params = optax.apply_updates(meta_params, updates)
        self.meta_agent.update_params(new_params)  # type: ignore

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

        # TODO: create dynamic environment make method - not all are Atari
        self.envs = make_atari_env(self.env_name, self.config.num_vec_envs)

        self.logger = logger
        self.writer_name = writer_name
        self.jit_compile = jit_compile

        action_space: gym.spaces.Discrete = self.envs.single_action_space  # type: ignore
        obs_space: gym.spaces.Box = self.envs.single_observation_space  # type: ignore

        self.n_actions = action_space.n.item()
        self.key, agent_key, target_key, value_key, buffer_key = jax.random.split(
            key, 5
        )

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

        self.buffer = MixedBuffer(buffer_key, config=self.config.buffer)

        # Init agent state
        current_obs, _ = self.envs.reset()

        self.state: AgentTrainerState = AgentTrainerState.create(
            self.policy_agent.get_params(),
            self.target_agent.get_params(),
            self.value_agent.get_params(),
            self._create_optim(),
            self._create_optim(),
            self.config.ema_decay,
            self.config.ema_eps,
            current_obs=current_obs,
        )

        # Checkpointing
        self.cp_manager = CheckpointManager(
            self.env_name,
            config=self.config.checkpoint,
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
            (self.config.num_vec_envs, *obs_space.shape),
            dtype=obs_space.dtype,
        )
        _ = self.policy_agent(dummy_obs)
        _ = self.target_agent(dummy_obs)
        _ = self.value_agent(dummy_obs)

        # Warm buffer
        self.collect()
        self.buffer.clear()

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

    def _compute_policy_loss(
        self,
        batch: BufferSamples,
        targets: DiscoAgentOutput,
    ) -> Tuple[chex.Array, AgentLosses]:
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
        total_loss : chex.Array
            Total loss
        losses : AgentLosses
            An object containing the agent losses
        """
        pi_loss = compute_kl_loss(targets.pi, batch.preds.pi).mean()
        y_loss = compute_kl_loss(targets.y, batch.preds.y).mean()
        z_loss = compute_z_loss(batch.preds.z, targets.z, batch.actions).mean()
        aux_pi_loss = compute_aux_policy_loss(
            batch.preds.aux_pi,
            batch.preds.pi,
            batch.actions,
            batch.discounts,
        ).mean()

        losses = AgentLosses(
            pi=pi_loss,
            y=y_loss,
            z=z_loss,
            aux_pi=aux_pi_loss,
        ).compute_total(self.config.loss_costs)

        return losses.total, losses

    def _update_policy_step(
        self,
        batch: BufferSamples,
        targets: DiscoAgentOutput,
    ) -> None:
        """
        Performs a single policy update step.

        Parameters
        ----------
        batch : BufferSamples
            Batch of experience from the buffer
        targets : DiscoAgentOutput
            Targets generated by the meta-network `(π̂, ŷ, ẑ)`
        """
        params = self.state.params.policy

        def loss_fn(p) -> Tuple[chex.Array, AgentLosses]:
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

        # Log metrics
        losses: AgentLosses = losses
        metrics = {
            **losses.to_metrics(),
            "inner/grad_norm": optax.global_norm(grads),
            "inner/steps_trained": self.state.steps_trained,
        }
        self.log(metrics)

    def step(self, targets: DiscoAgentOutput) -> None:
        """
        Perform a single training step.

        Parameters
        ----------
        targets : DiscoAgentOutput
            Targets from the meta-network
        """
        batch = self.buffer.sample(self.config.batch_size)
        self._update_policy_step(batch, targets)

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
        self.value_agent.update_params(self.state.value.params)

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

            values, value_h_state = self.value_agent(
                current_obs,
                h_state=hidden.value,
            )

            # Env step
            actions = self.policy_agent.act(preds.pi)
            next_obs, rewards, discounts, info = self.envs.step(actions)

            # Accumulate step data and episode stats
            collect_state = collect_state.append_step(
                actions, rewards, discounts, values, preds, target_preds
            )
            collect_state = collect_state.record_episodes(info)

            # Update and reset hidden states on episode boundaries
            hidden.update(h_state, target_h_state, value_h_state)
            hidden.reset_on_done(discounts, self.n_actions)

            current_obs = next_obs

        # Update state with new env state
        self.state.update_hidden_states(hidden)
        self.state.update_obs(current_obs)

        # Add to buffer
        batched_samples = collect_state.to_batches()
        self.buffer.add(**batched_samples.to_dict())

        # Log episodic metrics
        if metrics := collect_state.episode_metrics():
            self.log(metrics)

        return collect_state

    def get_values(self, batch: BufferSamples) -> DiscoValueOutputs:
        """
        Computes state-value function predictions for meta-gradient computation.

        Parameters
        ----------
        batch : BufferSamples
            Batch of experience from buffer

        Returns
        -------
        value_outs : DiscoValueOutputs
            Value function outputs
        """
        value_state = self.state.value

        # Sync params to network
        self.value_agent.update_params(value_state.params)

        # Transpose to (T, B) for V-trace
        batch = batch.to_time_first()
        discounts = batch.discounts * self.config.value.gamma

        # [:-1] = Drop last timestep
        rho = compute_importance_weights(
            batch.preds.pi[:-1],  # type: ignore
            batch.target_preds.pi[:-1],  # type: ignore
            batch.actions[:-1],  # type: ignore
        )

        value_targets, advantages = compute_vtrace(
            batch.values,
            batch.rewards[:-1],  # type: ignore
            discounts[:-1],  # type: ignore
            self.config.value.td_lambda,
            rho,
        )

        td = value_targets - batch.values[:-1]  # type: ignore

        # Compute EMAs
        norm_adv = value_state.adv_ema.update_and_normalize(advantages)
        norm_td = value_state.td_ema.update_and_normalize(td, subtract_mean=False)

        # Update state
        self.state.update_value_state(value_state)

        return DiscoValueOutputs(
            value=batch.values,
            value_targets=value_targets,
            advantages=advantages,
            normalized_advantages=norm_adv,
            td=td,
            normalized_td=norm_td,
            rho=rho,
        )

    def update_value_params(self, batch: BufferSamples) -> DiscoValueOutputs:
        """
        Update value function parameters.

        Parameters
        ----------
        batch : BufferSamples
            Batch of experience from buffer

        Returns
        -------
        values_outs : DiscoValueOutputs
            Value function outputs
        """
        v_state = self.state.value

        def value_loss_fn(params) -> Tuple[chex.Array, DiscoValueOutputs]:
            """Compute value loss."""
            self.value_agent.update_params(params)
            value_outs = self.get_values(batch)

            net_out = value_outs.value[:-1]  # type: ignore

            # Value loss from normalized TD
            # loss = 0.5 * (value - stop_grad[value + TD])^2
            value_target = jax.lax.stop_gradient(net_out + value_outs.normalized_td)
            value_loss = 0.5 * jnp.square(net_out - value_target).mean()
            value_loss = self.config.loss_costs.value * value_loss

            return value_loss, value_outs

        # Compute gradients
        (v_loss, v_outs), grads = jax.value_and_grad(value_loss_fn, has_aux=True)(
            v_state.params
        )

        # Update state
        v_state.apply_gradients(grads)
        self.state.update_value_state(v_state)

        # Sync params to network
        self.value_agent.update_params(v_state.params)

        # Log metrics
        v_outs: DiscoValueOutputs = v_outs
        metrics = {
            "value/loss": v_loss,
            **v_outs.to_metrics(),
            "value/grad_norm": optax.global_norm(grads),
        }
        self.log(metrics)

        return v_outs

    def save_checkpoint(self, force: bool = False) -> bool:
        """
        Save current state to checkpoint.

        Parameters
        ----------
        force : bool
            Force save even if within save interval. Default is `False`

        Returns
        -------
        saved : bool
            Whether checkpoint was actually saved
        """
        return self.cp_manager.save(
            self.state.steps_trained,
            self.state,
            force=force,
        )

    def load_checkpoint(self, step: int | None = None) -> bool:
        """
        Restore state from a checkpoint.

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
        abstract_state: AgentTrainerState = jax.tree.map(
            ocp.utils.to_shape_dtype_struct,
            self.state,
        )

        restored_state: AgentTrainerState = self.cp_manager.restore(
            step, abstract_state
        )  # type: ignore

        if restored_state is None:
            return False

        # Update state
        self.state = restored_state

        # Sync params to agent objects
        self.policy_agent.update_params(self.state.params.policy)
        self.target_agent.update_params(self.state.params.target)
        self.value_agent.update_params(self.state.value.params)

        return True

    def close(self) -> None:
        """
        Clean up resources.

        Ensures all async checkpoint operations complete and closes
        the checkpoint manager.
        """
        self.cp_manager.close()
        self.envs.close()
