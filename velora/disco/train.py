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
from velora.disco.config.state import AgentTrainerState, CollectState, RuleTrainerState
from velora.disco.ema import MovingAverage
from velora.disco.outputs import (
    AgentLossAux,
    AgentLosses,
    DiscoAgentOutput,
    LossStatistics,
    MetaInnerStepCarry,
    MetaLossAux,
    ValueOutputs,
)
from velora.disco.rollouts import Rollout
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

    `AgentTrainer` handles the inner loop of target rule learning (DiscoRL): collecting trajectories, generating targets via the target agent, and
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

        # Set values for loss_fn
        encoding, actions, discounts = (
            rollout.preds.encoding,
            rollout.actions,
            rollout.discounts,
        )

        # Compute policy gradients
        def loss_fn(params) -> Tuple[chex.Array, AgentLosses]:
            fresh_preds = self.policy_agent.functional_forward(encoding, params)

            return compute_policy_loss(
                targets,
                fresh_preds.pi,
                fresh_preds.y,
                fresh_preds.z,
                fresh_preds.aux_pi,
                actions,
                discounts,
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

    def update_value(
        self,
        rollout: Rollout,
        *,
        value_params: optax.Params | None = None,
        value_opt_state: optax.OptState | None = None,
    ) -> Tuple[ValueOutputs, optax.Params, optax.OptState]:
        """
        Update value network and return the new parameters and optimizer state.

        Can be used functionally with explicit `value_params` and `value_opt_state`.

        Parameters
        ----------
        rollout : Rollout
            Trajectory to compute value targets on
        value_params : optax.Params (optional)
            Value agent parameters for functional use. Default is `None`
        value_opt_state : optax.OptState (optional)
            Value agent optimizer state for functional use. Default is `None`

        Returns
        -------
        values_outs : ValueOutputs
            Value function outputs
        value_params : optax.Param
            Updated value agent parameters
        value_opt_state : optax.OptState
            Updated value agent optimizer state

        Raises
        ------
        func_error : ValueError
            Invalid parameters passed for functional mode
        """
        functional = value_params is not None

        if (value_params is None) != (value_opt_state is None):
            raise ValueError(
                f"'functional' mode enabled. Requires 'value_params' and 'value_opt_state' values.\nGot: {value_params=}, {value_opt_state=}"
            )

        params = value_params if functional else self.value_agent.get_params()
        opt_state = value_opt_state if value_opt_state else self.state.value_opt_state

        # Snapshot EMA state
        adv_ema = self.state.adv_ema
        td_ema = self.state.td_ema

        def loss_fn(p) -> Tuple[chex.Array, AgentLossAux]:
            """Compute value loss."""
            value_outs, new_adv_ema, new_td_ema = compute_value_outputs(
                rollout,
                self.ema_utils,
                adv_ema,
                td_ema,
                self.config.value.gamma,
                self.config.value.td_lambda,
            )
            net_out = value_outs.value[:-1]  # type: ignore

            # Value loss from normalized TD
            # loss = 0.5 * (value - stop_grad[value + TD])^2
            value_target = jax.lax.stop_gradient(net_out + value_outs.normalized_td)
            value_loss = (
                self.config.loss_costs.value
                * 0.5
                * jnp.square(net_out - value_target).mean()
            )

            aux = AgentLossAux(
                value_outs=value_outs,
                adv_ema=new_adv_ema,
                td_ema=new_td_ema,
            )
            return value_loss, aux

        # Compute gradients
        (v_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Apply optimizer
        updates, new_opt_state = self.value_optim.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Update state
        aux: AgentLossAux = aux

        if not functional:
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

        return aux.value_outs, new_params, new_opt_state

    def collect(self, seq_len: int | None = None) -> Rollout:
        """
        Collects a trajectory by interacting with the environment.

        Parameters
        ----------
        seq_len : int (optional)
            Trajectory length. Default is `None`.
            When `None` uses `config.seq_len`

        Returns
        -------
        rollout : Rollout
            Fresh trajectory of experience
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
            values, h_value = self.value_agent(obs, h_state=hidden.value)

            # Env step
            actions = self.policy_agent.act(np.asarray(preds.pi))
            actions_np = np.asarray(actions, dtype=np.int32)
            next_obs, rewards, terminated, truncated, info = self.envs.step(
                actions_np.squeeze()
            )

            discounts = np.where(
                terminated | truncated,
                np.float32(0.0),
                np.float32(1.0),
            )[:, None]

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

    def collect_stack(self, n_rollouts: int) -> Rollout:
        """
        Collect a stack of rollouts.

        Parameters
        ----------
        n_rollouts : int
            Number of rollouts to collect

        Returns
        -------
        stack : Rollout
            Stacked rollouts
        """
        return Rollout.from_list([self.collect() for _ in range(n_rollouts)])

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
    seed : int (optional)
        Random number generator seed. Default is `42`
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Recommended `True` to reduce training
        speed. Default is `True`
    cache_dir : str | None (optional)
        Directory path for JAX's persistent XLA compilation cache. On repeated runs with the same model shapes, compilation is skipped and loaded from disk instead. Set to `None` to disable. Only active when `jit_compile=True`. Default is `".cache/jax"`
    """

    def __init__(
        self,
        envs: EnvSet | EnvGroup,
        *,
        config: RuleTrainerSettings,
        seed: int = 42,
        jit_compile: bool = True,
        cache_dir: str | None = ".cache/jax",
    ) -> None:
        if jit_compile and cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", cache_dir)

        if isinstance(envs, EnvGroup):
            envs = EnvSet(envs)

        config.verify_params()
        envs.verify_packages()

        self._env_specs = envs.as_list()
        self.env_names = [name for name, _ in self._env_specs]
        self.num_envs = len(self._env_specs)

        self.config = config
        self.jit_compile = jit_compile
        self.cache_dir = cache_dir

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
            self.config.num_vec_envs,
            *self.meta_agent.hidden_sizes,
        )
        self.trainers: List[AgentTrainer] = []

        # Checkpointing
        self.cp_manager = CheckpointManager(self.config.checkpoint)
        self.rule_path = Path(self.cp_manager.cp_dir, "final_disco").resolve()

        # init console dashboard
        self.console = DiscoConsoleDashboard(
            self.config.console_config(
                envs=envs.env_categories(),
                params=self._dummy_params(trainer_keys[0]),
                complete_path=str(self.rule_path),
                jit_compile=self.jit_compile,
                cache_dir=self.cache_dir,
            )
        )

        # Initial setup
        self._initial_setup(trainer_keys)

    def _dummy_params(self, rng_key: chex.PRNGKey) -> DiscoParamsSettings:
        """
        Initializes a dummy set of agents to get their parameter counts.

        Parameters
        ----------
        rng_key : chex.PRNGKey
            Dummy random number generator key

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

    def _initial_setup(self, trainer_keys: List[chex.PRNGKey]) -> None:
        """
        Performs an initial forward pass through all agent networks and trainers
        and setups up environments.

        Includes -
            1. Materializing parameters before optimizer init
            2. JIT compile
            3. Creates environments

        Parameters
        ----------
        trainer_keys : List[chex.PRNGKey]
            List of trainer random number generated keys
        """
        self.console.start_setup((2 * self.num_envs) + 1)

        # Create remaining trainers with progress updates
        for i, (env_name, make_fn) in enumerate(self._env_specs):
            trainer = AgentTrainer(
                env_name,
                self.config.agent_trainer_config(),
                key=trainer_keys[i],
                logger=self.logger,
                writer_name=f"envs/{env_name}",
                make_fn=make_fn,
                jit_compile=self.jit_compile,
            )
            self.trainers.append(trainer)
            self.console.update_setup()

        dummy_rollout = self.trainers[0].collect()
        _ = self.meta_agent(dummy_rollout)
        self.console.update_setup()

        # Warm all trainers
        for trainer in self.trainers:
            trainer.warm()
            self.console.update_setup()

        self.console.finish_setup()

    def train(self) -> None:
        """
        Performs meta-training loop to discover an RL update rule.

        Uses the environment scheduler to rotate through environments in batches,
        accumulating meta-gradients in a running sum for efficiency.

        Includes -
        1. For each meta-step:

            a. Iterate through all agent trainers sequentially
            b. For each trainer, collect rollouts and perform inner loop updates
            c. Accumulate meta-gradients through the learning process
            d. Average gradients across all environments and update meta-network

        2. Log metrics and checkpoints periodically
        """
        self.console.start_training()

        for step in range(self.config.n_steps):
            accumulated_grad = jax.tree.map(
                jnp.zeros_like,
                self.meta_agent.get_params(),
            )
            all_rewards: List[float] = []
            all_losses: List[LossStatistics] = []

            # Iterate through each environment
            for idx, trainer in enumerate(self.trainers):
                train_rollouts = trainer.collect_stack(self.config.n_updates)
                valid_rollout = trainer.collect(self.config.seq_len * 2)

                # Compute meta_gradient for this agent
                meta_grad, disco_h, meta_h, losses = self._compute_meta_gradient(
                    idx,
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
                all_rewards.extend(trainer.get_episode_rewards())
                all_losses.append(losses)

                # Update hidden states
                self.state = self.state.update_hidden(idx, disco_h, meta_h)  # type: ignore

                self.console.update_progress()

            # Apply average gradients
            avg_grad = jax.tree.map(lambda g: g / self.num_envs, accumulated_grad)
            self._apply_meta_update(avg_grad)

            # Compute reward statistics
            reward_stats = RewardStatistics.from_rewards(all_rewards)
            loss_stats = LossStatistics.from_losses(all_losses)

            # Log metrics
            metrics = {
                "meta/grad_norm": float(optax.global_norm(avg_grad)),
                "meta/avg_reward": reward_stats.avg_reward,
                "meta/reward_std": reward_stats.reward_std,
                "meta/reward_min": reward_stats.reward_min,
                "meta/reward_max": reward_stats.reward_max,
                "meta/step": step,
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
        train_rollouts: Rollout,
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
        train_rollouts : Rollout
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
        from velora.disco.utils.mixflow import fwdrev_value_and_grad

        meta_params = self.meta_agent.get_params()
        policy_params = trainer.policy_agent.get_params()
        value_params = trainer.value_agent.get_params()
        disco_h, meta_h = self.state.hidden.get(trainer_idx)
        inner_grad_fn = fwdrev_value_and_grad

        def meta_loss_fn(meta_params) -> Tuple[chex.Array, MetaLossAux]:
            """
            Meta-loss function.

            Runs inner loop with MixFlow-MG reparameterization, then
            computes policy gradient on validation data.

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
                carry: MetaInnerStepCarry,
                rollout: Rollout,
            ) -> Tuple[MetaInnerStepCarry, Tuple[DiscoAgentOutput, chex.Array]]:
                """
                Single inner loop update step (MixFlow-MG reparameterized).

                Parameters
                ----------
                carry : MetaInnerStepCarry
                    Carry values
                rollout : Rollout
                    A trajectory of experience

                Returns
                -------
                new_carry : MetaInnerStepCarry
                    Carry values
                targets : Tuple[DiscoAgentOutput, chex.Array]
                    - Disco agent predictions
                    - Policy target pi predictions
                """
                # Generate targets from disco agent
                targets, d_h, m_h = self.meta_agent(
                    rollout,
                    disco_h_state=carry.disco_h,
                    meta_h_state=carry.meta_h,
                    params=meta_params,
                )

                encoding = rollout.preds.encoding
                actions, discounts = rollout.actions, rollout.discounts

                # Policy update
                def inner_policy_loss(
                    p, enc, tgt, act, disc
                ) -> Tuple[chex.Array, AgentLosses]:
                    new_preds = trainer.policy_agent.functional_forward(enc, p)

                    return compute_policy_loss(
                        tgt,
                        new_preds.pi,
                        new_preds.y,
                        new_preds.z,
                        new_preds.aux_pi,
                        act,
                        disc,
                        self.config.loss_cost,
                    )

                (_, _), p_grads = inner_grad_fn(inner_policy_loss, has_aux=True)(
                    carry.p_params, encoding, targets, actions, discounts
                )

                # Apply inner update
                p_updates, new_p_opt = trainer.policy_optim.update(
                    p_grads,
                    carry.p_opt_state,
                    carry.p_params,
                )
                new_p_params = optax.apply_updates(carry.p_params, p_updates)

                # Value update
                _, new_v_params, new_v_opt = trainer.update_value(
                    rollout,
                    value_params=carry.v_params,
                    value_opt_state=carry.v_opt_state,
                )

                new_carry = MetaInnerStepCarry(
                    p_params=new_p_params,
                    v_params=new_v_params,
                    disco_h=d_h,
                    meta_h=m_h,
                    p_opt_state=new_p_opt,
                    v_opt_state=new_v_opt,
                )
                return new_carry, (targets, rollout.target_preds.pi)

            # Run inner loop
            init_carry = MetaInnerStepCarry(
                p_params=policy_params,
                v_params=value_params,
                disco_h=disco_h,
                meta_h=meta_h,
                p_opt_state=trainer.state.policy_opt_state,
                v_opt_state=trainer.state.value_opt_state,
            )

            final_out, (all_targets, all_targets_pi) = jax.lax.scan(
                _inner_step,
                init_carry,  # type: ignore
                train_rollouts,  # type: ignore
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
            adv = jax.lax.stop_gradient(value_outs.normalized_advantages)

            pg_loss = compute_policy_gradient_loss(
                valid_rollout.preds.pi,
                valid_rollout.actions,
                adv,
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
                disco_h=final_out.disco_h,  # type: ignore
                meta_h=final_out.meta_h,  # type: ignore
                p_params=final_out.p_params,
                value_outs=value_outs,
                v_params=final_out.v_params,
                v_opt_state=final_out.v_opt_state,
            )
            return meta_loss, aux

        # Compute gradients w.r.t meta-network state (parameters)
        (meta_loss, aux), meta_grad = jax.value_and_grad(meta_loss_fn, has_aux=True)(
            meta_params
        )
        aux: MetaLossAux = aux

        # Update params and value state
        trainer.policy_agent.update_params(aux.p_params)  # type: ignore
        trainer.target_agent.update_params(aux.p_params)  # type: ignore

        trainer.value_agent.update_params(aux.v_params)  # type: ignore
        trainer.state = trainer.state.update_value_opt(aux.v_opt_state)

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

    def _apply_meta_update(self, avg_grad: chex.ArrayTree) -> None:
        """
        Apply meta-gradient update to meta-network parameters.

        Parameters
        ----------
        avg_grad : chex.ArrayTree
            Averaged gradients across all agents
        """
        meta_params = self.meta_agent.get_params()

        updates, new_opt_state = self.meta_optim.update(
            avg_grad,
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

        # Reset all agent trainers
        for idx in range(self.num_envs):
            self.reset_trainer(idx)

        return True

    def reset_trainer(self, trainer_idx: int) -> None:
        """
        Reset a trainer to initial state.

        Parameters
        ----------
        trainer_idx : int
            Index of the trainer to reset
        """
        env_name, make_fn = self._env_specs[trainer_idx]

        # Close existing trainer
        self.trainers[trainer_idx].close()

        # Fresh RNG keys
        self.key, new_key = jax.random.split(self.key, 2)

        # Init new trainer
        self.trainers[trainer_idx] = AgentTrainer(
            env_name,
            self.config.agent_trainer_config(),
            key=new_key,
            logger=self.logger,
            writer_name=f"envs/{env_name}",
            make_fn=make_fn,
            jit_compile=self.jit_compile,
        )

        # Reset hidden state for this environment
        self.state: RuleTrainerState = self.state.update_hidden(trainer_idx, None, None)

    def close(self) -> None:
        """Clean up resources."""
        self.cp_manager.close()

        for trainer in self.trainers:
            trainer.close()

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
