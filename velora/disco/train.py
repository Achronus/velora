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

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from velora.base.outputs import RewardStatistics
from velora.cli.disco.dashboard import DiscoConsoleDashboard
from velora.cli.disco.settings import DiscoParamsSettings
from velora.disco.agent import DiscoAgent, DiscoValueAgent, PolicyAgent
from velora.disco.config.settings import AgentTrainerSettings, RuleTrainerSettings
from velora.disco.config.state import (
    AgentSnapshot,
    AgentTrainerState,
    CollectState,
    RuleTrainerState,
)
from velora.disco.ema import MovingAverage
from velora.disco.outputs import (
    AgentLossAux,
    AgentLosses,
    DiscoAgentOutput,
    LossStatistics,
    MetaLossAux,
    ValueOutputs,
)
from velora.disco.rollouts import Rollout, RolloutStack
from velora.disco.scheduler import EnvironmentScheduler, SchedulerSettings
from velora.disco.utils.compute import compute_value_outputs
from velora.disco.utils.loss import (
    compute_entropy_loss,
    compute_meta_reg_loss,
    compute_policy_gradient_loss,
    compute_policy_loss,
)
from velora.gym.envs import EnvGroup, EnvSet, MakeFn
from velora.nn.optim import scale_by_adan_no_denom
from velora.tracking.logger import MetricsLogger
from velora.tracking.manager import CheckpointManager


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
    make_fn : MakeFn
        Factory function to create the environment
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
        make_fn: MakeFn,
        jit_compile: bool = False,
    ) -> None:
        self.config = config
        self.env_name = env_name

        self.envs = make_fn(self.env_name, self.config.num_vec_envs)

        self.logger = logger
        self.writer_name = writer_name
        self.jit_compile = jit_compile

        self.action_space: gym.spaces.Discrete = self.envs.single_action_space  # type: ignore
        self.obs_space: gym.spaces.Box = self.envs.single_observation_space  # type: ignore

        self.n_actions = self.action_space.n.item()
        self.key, agent_key, target_key, value_key = jax.random.split(key, 4)

        self.policy_agent = PolicyAgent(
            self.obs_space,
            self.action_space,
            config=self.config.agent,
            key=agent_key,
            jit_compile=self.jit_compile,
        )

        self.target_agent = PolicyAgent(
            self.obs_space,
            self.action_space,
            config=self.config.agent,
            key=target_key,
            jit_compile=self.jit_compile,
        )

        self.value_agent = DiscoValueAgent(
            self.obs_space,
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

        # Episode reward tracking
        self._episode_rewards: Tuple[float, ...] = ()

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

    def warm(self) -> None:
        """
        Performs an initial forward pass through the agent networks.

        Includes -
            1. Materializing parameters before optimizer init
            2. JIT compile
        """
        dummy_obs = jnp.zeros(
            (self.config.num_vec_envs, *self.obs_space.shape),
            dtype=self.obs_space.dtype,
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

        def loss_fn(params) -> Tuple[chex.Array, AgentLossAux]:
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

            aux = AgentLossAux(
                value_outs=value_outs,
                adv_ema=adv_ema,
                td_ema=td_ema,
            )
            return value_loss, aux

        # Compute gradients
        (v_loss, aux), grads = jax.value_and_grad(
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
        aux: AgentLossAux = aux
        self.value_agent.update_params(new_params)  # type: ignore
        self.state = self.state.update_value_opt(new_opt_state)
        self.state = self.state.update_ema(aux.adv_ema, aux.td_ema)

        # Log metrics
        metrics = {
            "value/loss": v_loss,
            **aux.value_outs.to_metrics(),
            "value/grad_norm": optax.global_norm(grads),
        }
        self.log(metrics)

        return aux.value_outs

    def collect(self, seq_len: int | None = None) -> Rollout:
        """
        Collects a trajectory by interacting with the environment.

        Parameters
        ----------
        seq_len : int (optional)
            Trajectory length. Default is `None`.
            When `None` uses config sequence length

        Returns
        -------
        rollout : Rollout
            A single trajectory
        """
        seq_len = seq_len or self.config.seq_len
        obs = self.state.current_obs
        hidden = self.state.hidden
        collect_state = CollectState()

        # Trajectory collection
        for _ in range(seq_len):
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

        # Log episodic metrics and store rewards
        self._episode_rewards = collect_state.completed_returns

        if metrics := collect_state.episode_metrics():
            self.log(metrics)

        return collect_state.to_rollout()

    def collect_stack(self, n_rollouts: int) -> RolloutStack:
        """
        Collect a stack of rollouts.

        Parameters
        ----------
        n_rollouts : int
            Number of rollouts to collect

        Returns
        -------
        stack : RolloutStack
            Stacked rollouts
        """
        rollouts = [self.collect() for _ in range(n_rollouts)]
        return RolloutStack.from_list(rollouts)

    def get_episode_rewards(self) -> Tuple[float, ...]:
        """
        Get episode rewards from the most recent collection.

        Returns
        -------
        rewards : Tuple[float, ...]
            Episode rewards from last `collect()` call
        """
        return self._episode_rewards

    def close(self) -> None:
        """Clean up resources."""
        self.envs.close()

    def serialize(self, device: str | None = None) -> AgentSnapshot:
        """
        Serialize the trainer's full state for CPU-resident storage.

        Parameters
        ----------
        device : str (optional)
            Target device for arrays. If `None`, uses `jax.devices("cpu")[0]`.
            Default is `None`

        Returns
        -------
        snapshot : AgentSnapshot
            CPU-resident snapshot of this trainer's state
        """
        device = jax.devices(device or "cpu")[0]

        return AgentSnapshot(
            policy_params=jax.device_put(self.policy_agent.get_params(), device),
            target_params=jax.device_put(self.target_agent.get_params(), device),
            value_params=jax.device_put(self.value_agent.get_params(), device),
            trainer_state=jax.device_put(self.state, device),
        )


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
    envs : EnvSet | EnvGroup
        Environment set or group to use for rule discovery
    config : RuleTrainerSettings
        Configuration for meta-training
    scheduler_config : SchedulerConfig (optional)
        Configuration for the environment scheduler. When `None` uses `SchedulerConfig()`.
        Default is `None`
    seed : int (optional)
        Random number generator seed. Default is `42`
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Recommended `True` to reduce training
        speed. Default is `True`
    """

    def __init__(
        self,
        envs: EnvSet | EnvGroup,
        *,
        config: RuleTrainerSettings,
        scheduler_config: SchedulerSettings | None = None,
        seed: int = 42,
        jit_compile: bool = True,
    ) -> None:
        if isinstance(envs, EnvGroup):
            envs = EnvSet(envs)

        config.verify_params()
        envs.verify_packages()

        self._env_specs = envs.as_list()
        self.env_names = [name for name, _ in self._env_specs]
        self.num_envs = len(self._env_specs)

        self.config = config
        self.jit_compile = jit_compile

        # Init logger
        self.logger = MetricsLogger(self.config.logger)
        self.logger.add_writer("meta")

        for name in self.env_names:
            self.logger.add_writer(f"envs/{name}")

        # Configure RNG keys
        key = jax.random.key(seed)
        self.key, meta_key, setup_key = jax.random.split(key, 3)

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
            self.config.num_vec_envs,
            *self.meta_agent.hidden_sizes,
        )

        # Init scheduler
        self.scheduler = EnvironmentScheduler(
            envs.as_specs(),
            config=scheduler_config or SchedulerSettings(),
            seed=seed,
        )

        # Checkpointing
        self.cp_manager = CheckpointManager(self.config.checkpoint)
        self.rule_path = Path(self.cp_manager.cp_dir, "final_disco").resolve()

        # init console dashboard
        self.console = DiscoConsoleDashboard(
            self.config.console_config(
                envs=envs.env_categories(),
                params=self._dummy_params(setup_key),
                batch_size=self.scheduler.config.batch_size,
                complete_path=str(self.rule_path),
                jit_compile=self.jit_compile,
            )
        )

        # Initial setup
        self._initial_setup(setup_key)

    def _dummy_params(self, rng_key: chex.PRNGKey) -> DiscoParamsSettings:
        """
        Initializes a dummy set of agents to get their parameter counts.

        Returns
        -------
        param_counts : DiscoParamsSettings
            Parameter counts for all agents
        """
        config = self.config.agent_trainer_config()
        env_name, make_fn = self._env_specs[0]
        envs = make_fn(env_name, self.config.num_vec_envs)

        policy = PolicyAgent(
            envs.single_observation_space,  # type: ignore
            envs.single_action_space,  # type: ignore
            config=config.agent,
            key=rng_key,
        )

        value = DiscoValueAgent(
            envs.single_observation_space,  # type: ignore
            config.agent.n_hidden,
            config=config.value,
            key=rng_key,
            sparsity=config.agent.sparsity,
        )
        envs.close()

        return DiscoParamsSettings(
            policy=policy.param_count,
            value=value.param_count,
            disco=self.meta_agent.param_count,
        )

    def _initial_setup(self, key: chex.PRNGKey) -> None:
        """
        Perform initial setup: create one trainer for warm-up and
        console dashboard initialization.

        The trainer is created temporarily, used for parameter counting
        and meta-agent warm-up, then torn down.

        Parameters
        ----------
        key : chex.PRNGKey
            Random number generator key
        """
        env_name, make_fn = self._env_specs[0]
        self.console.start_setup(3)

        # Create temp trainer for warm-up
        trainer = AgentTrainer(
            env_name,
            self.config.agent_trainer_config(),
            key=key,
            logger=self.logger,
            writer_name=f"warm/{env_name}",
            make_fn=make_fn,
            jit_compile=self.jit_compile,
        )
        self.console.update_setup()

        dummy_rollout = trainer.collect()
        _ = self.meta_agent(dummy_rollout)
        self.console.update_setup()

        # Init env state and clean up
        self.scheduler.set_state(0, trainer.serialize())
        trainer.close()

        self.console.update_setup()
        self.console.finish_setup()

    def _create_trainer(self, env_idx: int) -> AgentTrainer:
        """
        Create a fresh `AgentTrainer` for a specific environment.

        If the scheduler holds a serialized state for this environment, the trainer's state is restored from it.

        Parameters
        ----------
        env_idx : int
            Environment spec list index

        Returns
        -------
        trainer : AgentTrainer
            New trainer, optionally restored from serialized state
        """
        env_name, make_fn = self._env_specs[env_idx]

        # Deterministic key per env, varying by meta-step for resets
        trainer_key = jax.random.fold_in(self.key, env_idx + self.state.meta_step)

        trainer = AgentTrainer(
            env_name,
            self.config.agent_trainer_config(),
            key=trainer_key,
            logger=self.logger,
            writer_name=f"envs/{env_name}",
            make_fn=make_fn,
            jit_compile=self.jit_compile,
        )

        # Restore state (if available)
        state = self.scheduler.get_state(env_idx)
        if state is not None:
            EnvironmentScheduler.restore_trainer(trainer, state)

        return trainer

    def _teardown_trainer(self, env_idx: int, trainer: AgentTrainer) -> None:
        """
        Serialize a trainer's state and close its environment.

        Parameters
        ----------
        env_idx : int
            Environment index
        trainer : AgentTrainer
            Trainer to use
        """
        self.scheduler.set_state(env_idx, trainer.serialize())
        trainer.close()

    def train(self) -> None:
        """
        Performs meta-training loop to discover an RL update rule.

        Uses the environment scheduler to rotate through environments in batches,
        accumulating meta-gradients in a running sum for efficiency.

        Includes -
        1. For each meta-step:

            a. Select environments for this step (all or a subset)
            b. Iterate in environment batches
            c. For each environment in each batch:

                - Create trainer (restore state if needed)
                - Collect rollouts and compute meta-gradient
                - Accumulate gradient into running sum
                - Serialize state and teardown trainer

            d. Average accumulate gradients and update meta-network

        2. Log metrics and checkpoints periodically
        """
        self.console.start_training()

        for step in range(self.config.n_steps):
            # Running gradient accumulation
            accumulated_grad = jax.tree.map(
                jnp.zeros_like,
                self.meta_agent.get_params(),
            )

            n_envs_processed = 0
            all_rewards: List[float] = []
            all_losses: List[LossStatistics] = []

            batch = self.scheduler.next_batch()

            # Environment iteration
            for env_batch in self.scheduler.iterate_concurrent(batch):
                for env_idx in env_batch:
                    trainer = self._create_trainer(env_idx)

                    # Collect data
                    train_rollouts = trainer.collect_stack(self.config.n_updates)
                    valid_rollout = trainer.collect(self.config.seq_len * 2)

                    # Compute meta_gradient for this agent
                    meta_grad, disco_h, meta_h, losses = self._compute_meta_gradient(
                        env_idx,
                        trainer,
                        train_rollouts,
                        valid_rollout,
                    )

                    # Collect metrics
                    accumulated_grad = jax.tree.map(
                        lambda acc, g: acc + g,
                        accumulated_grad,
                        meta_grad,
                    )
                    n_envs_processed += 1

                    all_rewards.extend(trainer.get_episode_rewards())
                    all_losses.append(losses)

                    # Update hidden states
                    self.state = self.state.update_hidden(env_idx, disco_h, meta_h)  # type: ignore

                    # Update value function
                    _ = trainer.update_value(valid_rollout)

                    # Close trainer
                    self._teardown_trainer(env_idx, trainer)
                    self.console.update_progress()

            # Update meta-gradients across processed environments
            avg_meta_grad = jax.tree.map(
                lambda g: g / n_envs_processed,
                accumulated_grad,
            )
            self._apply_meta_update(avg_meta_grad)

            # Compute reward statistics
            reward_stats = RewardStatistics.from_rewards(all_rewards)
            loss_stats = LossStatistics.from_losses(all_losses)

            # Log metrics
            metrics = {
                "meta/grad_norm": float(optax.global_norm(avg_meta_grad)),
                "meta/avg_reward": reward_stats.avg_reward,
                "meta/reward_std": reward_stats.reward_std,
                "meta/reward_min": reward_stats.reward_min,
                "meta/reward_max": reward_stats.reward_max,
                "meta/step": step,
                "meta/envs_processed": n_envs_processed,
                "meta/epoch": self.scheduler.epoch,
            }
            self.logger.log("meta", step, metrics)

            self.console.update_stats(**reward_stats.to_dict())
            self.console.update_losses(**loss_stats.to_dict())

            # Checkpoint periodically
            if step % self.config.checkpoint.freq == 0:
                self.save_checkpoint()

            self.console.update_progress()

        # Final cleanup
        self.save_checkpoint(force=True)
        self.save_rule()

        self.close()
        self.console.finish_training()

    def _compute_meta_gradient(
        self,
        trainer_idx: int,
        trainer: AgentTrainer,
        train_rollouts: RolloutStack,
        valid_rollout: Rollout,
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree, LossStatistics]:
        """
        Compute meta-gradient for a single agent through the inner loop.

        Structured as a pure function for JIT compilation.

        Parameters
        ----------
        trainer_idx : int
            Index of the trainer
        trainer : AgentTrainer
            The agent trainer
        train_rollouts : RolloutStack
            Pre-collected training rollouts
        valid_rollout : Rollout
            Validation rollout for meta-loss

        Returns
        -------
        meta_grad : ArrayTree
            Gradients w.r.t. meta-network params
        disco_h : chex.Array
            Updated Disco network hidden state
        meta_h : chex.Array
            Updated Meta LNN hidden state
        losses : LossStatistics
            Trainer loss statistics
        """
        meta_params = self.meta_agent.get_params()
        policy_params = trainer.policy_agent.get_params()
        disco_h, meta_h = self.state.hidden.get(trainer_idx)

        def meta_loss_fn(meta_params) -> Tuple[chex.Array, MetaLossAux]:
            """
            Meta-loss function.

            Runs inner loop, then computes policy gradient on validation.

            Parameters
            ----------
            meta_params : optax.Params
                Meta agent parameters

            Returns
            -------
            meta_loss : chex.Array
                Meta loss
            aux : MetaLossAux
                Meta loss auxiliary values
            """

            def _inner_step(
                carry: Tuple[optax.Params, chex.Array, chex.Array, optax.OptState],
                rollout: Rollout,
            ) -> Tuple[
                Tuple[optax.Params, chex.Array, chex.Array, optax.OptState],
                Tuple[DiscoAgentOutput, chex.Array],
            ]:
                """
                Single inner loop update step.

                Parameters
                ----------
                carry : Tuple[optax.Params, chex.Array, chex.Array, optax.OptState]
                    - Policy parameters
                    - Disco network hidden state
                    - Meta LNN hidden state
                    - Policy optimizer state
                rollout : Rollout
                    A single trajectory of experience

                Returns
                -------
                new_carry : Tuple[optax.Params, chex.Array, chex.Array, optax.OptState]
                    - Updated policy parameters
                    - Updated Disco network hidden state
                    - Updated Meta LNN hidden state
                    - New policy optimizer state
                targets : Tuple[DiscoAgentOutput, chex.Array]
                    - Disco agent predictions
                    - Policy target pi predictions
                """
                p_params, disco_h, meta_h, opt_state = carry

                # Generate targets from disco agent
                targets, disco_h, meta_h = self.meta_agent(
                    rollout,
                    disco_h_state=disco_h,
                    meta_h_state=meta_h,
                )

                # Compute policy loss and gradient
                def inner_loss_fn(params) -> Tuple[chex.Array, AgentLosses]:
                    return compute_policy_loss(
                        targets,
                        rollout.preds.pi,
                        rollout.preds.y,
                        rollout.preds.z,
                        rollout.preds.aux_pi,
                        rollout.actions,
                        rollout.discounts,
                        self.config.loss_cost,
                    )

                (_), grads = jax.value_and_grad(inner_loss_fn, has_aux=True)(p_params)

                # Apply inner update
                updates, new_opt_state = trainer.policy_optim.update(
                    grads,
                    opt_state,
                    p_params,
                )
                new_policy_params = optax.apply_updates(p_params, updates)

                new_carry = (new_policy_params, disco_h, meta_h, new_opt_state)
                return new_carry, (targets, rollout.target_preds.pi)

            # Run inner loop
            init_carry = (
                policy_params,
                disco_h,
                meta_h,
                trainer.state.policy_opt_state,
            )
            (final_p_params, final_d_h, final_m_h, _), (all_targets, all_targets_pi) = (
                jax.lax.scan(
                    _inner_step,
                    init_carry,  # type: ignore
                    train_rollouts,  # type: ignore
                )
            )

            # Compute value outputs on validation rollout
            value_outs, _, _ = compute_value_outputs(
                valid_rollout,
                trainer.ema_utils,
                trainer.state.adv_ema,
                trainer.state.td_ema,
                trainer.config.value.gamma,
                trainer.config.value.td_lambda,
            )

            pg_loss = compute_policy_gradient_loss(
                valid_rollout.preds.pi,
                valid_rollout.actions,
                value_outs.normalized_advantages,
            ).mean()
            entropy_loss = compute_entropy_loss(
                valid_rollout.preds.pi,
                self.config.entropy_coef,
            )
            reg_loss = compute_meta_reg_loss(
                jax.tree.map(lambda x: x[-1], all_targets),  # last targets
                jax.tree.map(lambda x: x[-1], all_targets_pi),
                self.config.reg_scale,
                self.config.kl_reg,
            )
            meta_loss = pg_loss + entropy_loss + reg_loss

            aux = MetaLossAux(
                pg_loss=pg_loss,
                entropy_loss=entropy_loss,
                reg_loss=reg_loss,
                disco_h=final_d_h,
                meta_h=final_m_h,
                p_params=final_p_params,
                value_outs=value_outs,
            )
            return meta_loss, aux

        # Compute gradients w.r.t meta-network state (parameters)
        (meta_loss, aux), meta_grad = jax.value_and_grad(meta_loss_fn, has_aux=True)(
            meta_params
        )
        aux: MetaLossAux = aux

        # Update params
        trainer.policy_agent.update_params(aux.p_params)  # type: ignore
        trainer.target_agent.update_params(aux.p_params)  # type: ignore

        # Log metrics
        self._log_meta_metrics(
            trainer_idx,
            aux.pg_loss,
            aux.entropy_loss,
            aux.reg_loss,
            meta_loss,
            aux.value_outs,
        )

        losses = LossStatistics(
            meta=float(meta_loss),
            policy_gradient=float(aux.pg_loss),
            entropy=float(aux.entropy_loss),
            regularization=float(aux.reg_loss),
        )

        return meta_grad, aux.disco_h, aux.meta_h, losses

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
        self.logger.log(
            f"envs/{self.env_names[trainer_idx]}",
            self.state.meta_step,
            metrics,
        )

    def _apply_meta_update(self, meta_grad: chex.ArrayTree) -> None:
        """
        Apply meta-gradient update to meta-network parameters.

        Parameters
        ----------
        meta_grad : chex.ArrayTree
            Averaged gradients across all agents
        """
        meta_params = self.meta_agent.get_params()

        updates, new_opt_state = self.meta_optim.update(
            meta_grad,
            self.state.meta_opt_state,
            meta_params,
        )
        new_params = optax.apply_updates(meta_params, updates)
        self.meta_agent.update_params(new_params)  # type: ignore

        # Update state
        self.state = self.state.update_opt(new_opt_state)

    def save_checkpoint(self, force: bool = False) -> bool:
        """
        Save current state to checkpoint.

        Includes -
            - Meta agent parameters
            - Trainer state

        Parameters
        ----------
        force : bool (optional)
            Force save even if within save interval. Default is `False`

        Returns
        -------
        saved : bool
            Whether checkpoint was actually saved
        """
        if not force and not self.cp_manager.should_save(self.state.meta_step):
            return False

        checkpoint = {
            "meta_params": self.meta_agent.get_params(),
            "state": self.state,
        }

        self.cp_manager.save(self.state.meta_step, checkpoint, force=True)
        return True

    def load_checkpoint(self, step: int | None = None) -> bool:
        """
        Restore state from a checkpoint.

        On restore, all serialized agent states are reset to start fresh.

        Parameters
        ----------
        step : int (optional)
            Specific step to restore, or `None` for latest. Default is `None`

        Returns
        -------
        success : bool
            Whether restoration was successful
        """
        abstract_checkpoint = {
            "meta_params": jax.tree.map(
                ocp.utils.to_shape_dtype_struct,
                self.meta_agent.get_params(),
            ),
            "state": jax.tree.map(
                lambda x: ocp.utils.to_shape_dtype_struct(x) if x is not None else None,
                self.state,
            ),
        }

        restored = self.cp_manager.restore(step, abstract_checkpoint)

        if restored is None:
            return False

        # Restore meta-agent params and state
        self.meta_agent.update_params(restored["meta_params"])
        self.state = restored["state"]

        # Reset all serialized agent states (fresh start)
        for idx in range(self.num_envs):
            self.scheduler.set_state(idx, None)

        return True

    def close(self) -> None:
        """Clean up resources."""
        self.cp_manager.close()

    def save_rule(self) -> Path:
        """
        Export the discovered update rule as a standalone `DiscoAgent`.

        Saves the meta-agent's parameters and configuration so the rule
        can be loaded independently with `DiscoAgent.load()` for use
        in new environments and architectures.

        Returns
        -------
        path : Path
            Path where the rule was saved
        """
        root_dir, sub_dir = self.rule_path.parent, self.rule_path.name

        if self.rule_path.exists():
            shutil.rmtree(self.rule_path)

        return self.meta_agent.save(root_dir, sub_dir, timestamp=False)
