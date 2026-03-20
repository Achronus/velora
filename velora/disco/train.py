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

import json
import math
import queue
import random
import shutil
import threading
from pathlib import Path
from typing import Callable, Dict, List, Self, Tuple

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from velora.cli.disco.dashboard import DiscoConsoleDashboard
from velora.cli.disco.settings import DiscoParamsSettings
from velora.cli.disco.simple import SimpleDashboard
from velora.disco.agent import DiscoAgent, DiscoValueAgent, PolicyAgent
from velora.disco.config.settings import AgentTrainerSettings, RuleTrainerSettings
from velora.disco.config.state import AgentTrainerState, RuleTrainerState
from velora.disco.ema import EMAState, MovingAverage
from velora.disco.outputs import (
    AgentLossAux,
    LossStatistics,
    MetaGradOutput,
    MetaInnerStepCarry,
    MetaLossAux,
    MetaStepStats,
    ValueOutputs,
)
from velora.disco.rollouts import Rollout, RolloutBuffer
from velora.disco.utils.budget import sample_budget
from velora.disco.utils.compute import compute_value_outputs
from velora.disco.utils.loss import (
    compute_entropy_loss,
    compute_meta_reg_loss,
    compute_policy_gradient_loss,
    compute_policy_loss,
)
from velora.disco.utils.mixflow import fwdrev_value_and_grad
from velora.gym.envs import EnvGroup, EnvSet, MakeFn
from velora.nn.optim import scale_by_adan_no_denom
from velora.tracking.episode import EpisodeTracker
from velora.tracking.logger import MetricsLogger
from velora.tracking.manager import CheckpointManager
from velora.utils.config import dump_config, load_config
from velora.utils.format import cache_status
from velora.utils.transforms import stack_pytrees, unstack_pytree

_RESET_SENTINEL = object()
"""Sentinel pushed by actor thread to signal that trainer has
exhausted its lifetime budget and needs a main-thread reset
before next rollout can be collected."""


class AgentTrainer:
    """
    Trains a single `PolicyAgent` in a single environment using a learned update rule.

    `AgentTrainer` handles the inner loop of target rule learning (DiscoRL): collecting trajectories, generating targets via the target agent, and
    updating the `PolicyAgent` to minimize prediction error against those targets.

    This class is designed to be instantiated multiple times by `RuleTrainer`,
    with each instance training an agent in a different environment. During
    meta-training, it exposes differentiable update steps that allow gradients
    to flow back through the agent's learning process.

    Lifetime
    ---------
    Each trainer is assigned a step budget on construction, sampled from the
    DiscoRL distribution `{20M, 50M, 100M, 200M}` environment steps with
    weights inversely proportional to budget size. When the budget is
    exhausted, `needs_reset` becomes `True` and `RuleTrainer` will
    replace this trainer with a freshly initialized one before the next
    collection step. This ensures the meta-learner continually observes
    agents learning from scratch rather than converged policies.

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
    budget_rng : random.Random
        Python RNG used to sample the initial step budget. Passed in from
        `RuleTrainer` so all trainers share a single seeded RNG, keeping
        budget draws reproducible
    max_actions : int
        Maximum number of discrete actions across all environments in the
        training set
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    use_bfloat16 : bool (optional)
        Flag to set all floating-point arrays in the returned `Rollout` to
        `bfloat16` on GPU transfer. Halves VRAM usage for rollout buffers with
        negligible effect on training quality. `actions` remain `int32`.
        Default is `True`
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
        budget_rng: random.Random,
        max_actions: int,
        jit_compile: bool = False,
        use_bfloat16: bool = True,
    ) -> None:
        self.config = config
        self.env_name = env_name

        self.envs = make_fn(self.env_name, self.config.batch_size)

        self.logger = logger
        self.writer_name = writer_name
        self.jit_compile = jit_compile
        self.use_bfloat16 = use_bfloat16
        self.max_actions = max_actions

        self.action_space: gym.spaces.Discrete = self.envs.single_action_space  # type: ignore
        self.obs_space: gym.spaces.Box = self.envs.single_observation_space  # type: ignore

        self.n_actions = self.action_space.n.item()
        self.key, agent_key, target_key, value_key = jax.random.split(key, 4)

        self.policy_agent = PolicyAgent(
            self.obs_space,
            self.action_space,
            config=self.config.agent,
            key=agent_key,
            max_actions=self.max_actions,
            jit_compile=self.jit_compile,
        )

        self.target_agent = PolicyAgent(
            self.obs_space,
            self.action_space,
            config=self.config.agent,
            key=target_key,
            max_actions=self.max_actions,
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
        )  # (B, H)

        self.meta_optim: optax.GradientTransformation = None  # type: ignore
        self.meta_opt_state: optax.OptState = None  # type: ignore
        self.meta_optim_update: Callable = None  # type: ignore

        # Lifetime tracking
        self._env_steps: int = 0
        self._step_budget: int = sample_budget(budget_rng)

        # Episode tracking
        self.episode_tracker = EpisodeTracker(self.config.batch_size)
        self._collection_step = 0

        # Pre-allocated buffers
        self.train_buffer = RolloutBuffer(
            n_rollouts=self.config.n_updates,
            n_envs=self.config.batch_size,
            seq_len=self.config.seq_len,
            n_actions=self.max_actions,
            encoding_dim=self.policy_agent.encoding_dim,
            prediction_dim=self.config.agent.prediction_size,
            q_dim=self.config.agent.q_size,
            use_bfloat16=self.use_bfloat16,
        )
        self.valid_buffer = RolloutBuffer(
            n_rollouts=1,
            n_envs=self.config.batch_size,
            seq_len=self.config.seq_len * 2,
            n_actions=self.max_actions,
            encoding_dim=self.policy_agent.encoding_dim,
            prediction_dim=self.config.agent.prediction_size,
            q_dim=self.config.agent.q_size,
            use_bfloat16=self.use_bfloat16,
        )

    @property
    def needs_reset(self) -> bool:
        """
        Check whether this trainer has exhausted its lifetime budget.
        """
        return self._env_steps >= self._step_budget

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

    def _init_meta_optim(self, meta_params: optax.Params) -> None:
        """
        Initializes a per-trainer meta-gradient optimizer to stabilize gradient norms.

        Uses Adan optimizer (without denominator) with gradient clipping.

        Returns
        -------
        optim : optax.GradientTransformation
            Configured optimizer chain
        """
        self.meta_optim = optax.chain(
            scale_by_adan_no_denom(),
            optax.clip(self.config.agent.max_grad_norm),
        )
        self.meta_opt_state = self.meta_optim.init(meta_params)

        # JIT compile update fn (if needed)
        self.meta_optim_update = (
            jax.jit(self.meta_optim.update)
            if self.jit_compile
            else self.meta_optim.update
        )

    def warm(self) -> None:
        """
        Performs an initial forward pass through the agent networks.

        Includes -
            1. Materializing parameters before optimizer init
            2. JIT compile caching
        """
        dummy_obs = jnp.zeros(
            (self.config.batch_size, *self.obs_space.shape),
            dtype=self.obs_space.dtype,
        )
        _ = self.policy_agent(dummy_obs)
        _ = self.target_agent(dummy_obs)
        _ = self.value_agent(dummy_obs)

    def log(self, metrics: Dict[str, float], *, idx: int | None = None) -> None:
        """
        Log metrics to `MetricsLogger` if logger is available.

        Parameters
        ----------
        metrics : Dict[str, float]
            Mapping of metric names to scalar values
        idx : int (optional)
            Step index. Default is `None`
        """
        if self.logger:
            idx = idx if idx is not None else self.state.steps_trained
            self.logger.log(self.writer_name, idx, metrics)

    def update_value(
        self,
        rollout: Rollout,
        *,
        value_params: optax.Params | None = None,
        value_opt_state: optax.OptState | None = None,
        adv_ema: EMAState | None = None,
        td_ema: EMAState | None = None,
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
        adv_ema : chex.Array (optional)
            Advantage EMA state. When provided (functional mode), this value is
            used instead of reading from `self.state`, preventing it from being
            embedded as a constant in a JAX trace. Default is `None`
        td_ema : chex.Array (optional)
            TD-error EMA state. Same semantics as `adv_ema`. Default is `None`

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
        func_mode_error : ValueError
            Invalid parameters passed for functional mode
        """
        functional = value_params is not None

        if (value_params is None) != (value_opt_state is None):
            raise ValueError(
                f"'functional' mode enabled. Requires 'value_params' and 'value_opt_state' values.\nGot: {value_params=}, {value_opt_state=}"
            )

        params = value_params if functional else self.value_agent.get_params()
        opt_state = (
            value_opt_state
            if value_opt_state is not None
            else self.state.value_opt_state
        )
        adv_ema = adv_ema if adv_ema is not None else self.state.adv_ema
        td_ema = td_ema if td_ema is not None else self.state.td_ema

        def loss_fn(p, r, adv, td) -> Tuple[chex.Array, AgentLossAux]:
            """Compute value loss."""
            value_outs, new_adv_ema, new_td_ema = compute_value_outputs(
                r,
                self.ema_utils,
                adv,
                td,
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
        (v_loss, aux), grads = fwdrev_value_and_grad(loss_fn, has_aux=True)(
            params, rollout, adv_ema, td_ema
        )

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

    def compute_value_outs(
        self,
        rollout: Rollout,
        adv_ema: EMAState,
        td_ema: EMAState,
    ) -> Tuple[ValueOutputs, EMAState, EMAState]:
        """
        Compute value outputs and updated EMA states from a rollout.

        Parameters
        ----------
        rollout : Rollout
            Trajectory to compute value targets on
        adv_ema : EMAState
            Current advantage EMA state
        td_ema : EMAState
            Current TD-error EMA state

        Returns
        -------
        value_outs : ValueOutputs
            Value function outputs (targets, advantages, normalized advantages)
        adv_ema : EMAState
            Updated advantage EMA state
        td_ema : EMAState
            Updated TD-error EMA state
        """
        return compute_value_outputs(
            rollout,
            self.ema_utils,
            adv_ema,
            td_ema,
            self.config.value.gamma,
            self.config.value.td_lambda,
        )

    def get_vmap_inputs(
        self,
    ) -> Tuple[
        optax.Params,
        optax.Params,
        optax.OptState,
        optax.OptState,
        EMAState,
        EMAState,
    ]:
        """
        Return the trainer's current state as a flat tuple of pytress,
        read for stacking across a chunk with `vmap`.

        Returns
        -------
        p_params : optax.Params
            Policy network parameters
        v_params : optax.Params
            Value network parameters
        p_opt_state : optax.OptState
            Policy optimizer state
        v_opt_state : optax.OptState
            Value optimizer state
        adv_ema : EMAState
            Advantage EMA state
        td_ema : EMAState
            TD-error EMA state
        """
        return (
            self.policy_agent.get_params(),
            self.value_agent.get_params(),
            self.state.policy_opt_state,
            self.state.value_opt_state,
            self.state.adv_ema,
            self.state.td_ema,
        )

    def _add_to_buffer(self, buffer: RolloutBuffer, seq_len: int) -> None:
        """
        Collects a trajectory by interacting with the environment and writes
        it into the `buffer`.

        Parameters
        ----------
        buffer : RolloutBuffer
            Pre-allocated buffer to write into
        seq_len : int
            Number of environment steps to collect. Must match the `T`
            dimension the buffer was constructed with

        Returns
        -------
        rollout : Rollout
            Fresh trajectory of experience
        """
        obs = self.state.current_obs
        hidden = self.state.hidden

        # Trajectory collection
        for _ in range(seq_len):
            obs_jax = jnp.asarray(obs)

            preds, h_policy = self.policy_agent(
                obs_jax,
                ocm_h_state=hidden.policy_ocm,
                acm_h_state=hidden.policy_acm,
            )
            target_preds, h_target = self.target_agent(
                obs_jax,
                ocm_h_state=hidden.target_ocm,
                acm_h_state=hidden.target_acm,
            )
            values, h_value = self.value_agent(obs_jax, h_state=hidden.value)

            # Env step
            actions = self.policy_agent.act(np.asarray(preds.pi))
            actions_np = np.asarray(actions, dtype=np.int32)
            next_obs, rewards, terminated, truncated, _ = self.envs.step(
                actions_np.squeeze(-1)
            )

            discounts = jnp.where(
                terminated | truncated,
                jnp.float32(0.0),
                jnp.float32(1.0),
            )[:, None]

            # Store step data and episode stats
            buffer.write_step(
                actions_np,
                rewards[:, None],
                np.asarray(discounts),
                np.asarray(values),
                preds,
                target_preds,
            )
            self.episode_tracker.record(rewards, terminated, truncated)

            # Update and reset hidden states on episode boundaries
            hidden = hidden.update(h_policy, h_target, h_value)
            hidden = hidden.reset_on_done(discounts)

            obs = next_obs

        # Update trainer state
        self.state = self.state.update_hidden(hidden)
        self.state = self.state.update_obs(jnp.asarray(obs))

    def collect_stack(self, n_rollouts: int) -> Rollout:
        """
        Collect a stack of `N` rollout training trajectories.

        Writes each rollout directly into `train_buffer` and returns a single
        stacked `Rollout` with shape `(N, B, T, ...)`.

        Parameters
        ----------
        n_rollouts : int
            Number of rollouts to collect. Must match the `n_rollouts`
            dimension in the `train_buffer`

        Returns
        -------
        stack : Rollout
            Stacked rollouts with shape `(N, B, T, ...)`.
            All arrays are Jax arrays loaded onto the default device
        """
        self.episode_tracker.reset()

        for _ in range(n_rollouts):
            self._add_to_buffer(self.train_buffer, self.config.seq_len)
            self.train_buffer.next_rollout()

        rollout = self.train_buffer.to_rollout()

        # Accumulate training environment steps for lifetime tracking
        self._env_steps += n_rollouts * self.config.seq_len

        # Log episodic metrics and store rewards
        if metrics := self.episode_tracker.metrics():
            self.log(metrics, idx=self._collection_step)

        self._collection_step += 1
        return rollout

    def collect_valid(self) -> Rollout:
        """
        Collect a single validation trajectory using the `valid_buffer`.

        Returns
        -------
        rollout : Rollout
            Single rollout with shape `(1, seq_len * 2, ...)`.
            All arrays are JAX arrays loaded onto the default device
        """
        self._add_to_buffer(self.valid_buffer, self.config.seq_len * 2)
        self.valid_buffer.next_rollout()
        return self.valid_buffer.to_rollout()[0]

    def ep_return(self) -> float:
        """Windowed mean episodic return."""
        return self.episode_tracker.windowed_mean_return

    def ep_length(self) -> float:
        """Windowed mean episode length."""
        return self.episode_tracker.windowed_mean_length

    def close(self) -> None:
        """Clean up resources."""
        self.envs.close()


class RuleTrainer:
    """
    Discovers a Reinforcement Learning (RL) update rule by meta-training across
    a population of agents spanning diverse environments.

    `RuleTrainer` handles the outer loop of target rule learning (DiscoRL): managing
    a population of `AgentTrainer`s, computing meta-gradients from their learning
    progress, and updating the meta-network to improve collective agent performance.

    Trainers are grouped by action-space size so that `jax.vmap` can compute
    meta-gradients for an entire group in one batched accelerator (E.g., GPU) call.
    A background thread pre-collects rollouts for the next group while the accelerator
    processes the current group, overlapping CPU environment stepping with accelerator
    gradient computation.

    Follows a Sebulba actor infrastructure used in the paper
    [Podracer architectures for scalable Reinforcement Learning](https://arxiv.org/abs/2104.06272).

    Parameters
    ----------
    envs : EnvSet | EnvGroup
        Environment set or group to use for rule discovery
    config : RuleTrainerSettings
        Configuration for meta-training
    agents_per_env : int (optional)
        Number of independent agent trainers to instantiate per environment.
        For example, if we have 57 environments and 2 agents per environment we have
        114 agent trainers. Formula: `n_agents = agents_per_env * n_envs`.

        Each trainer has its own parameters, optimizer state, and lifetime
        budget, giving the meta-learner diverse gradient signals: agents at different
        stages of learning and with different random initializations. Default is `2`
    max_group_size : int (optional)
        Maximum number of trainers to `vmap` simultaneously. Higher values
        improve GPU utilization but increase VRAM usage. Reduce if
        out of memory (OOM). Default is `8`
    seed : int (optional)
        Random number generator seed. Default is `42`
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `True`
    cache_dir : str | None (optional)
        Directory path for JAX's persistent XLA compilation cache.
        Set to `None` to disable. Default is `".cache/jax"`

        On repeated runs with the same model shapes, compilation is skipped and
        loaded from disk instead. Only active when `jit_compile=True`.
    verbose : bool (optional)
        Dashboard display settings. Default is `True`.
            - If `True` - renders the full Rich dashboard
            - If `False` - uses a lightweight `tqdm` progress bar instead
    use_bfloat16 : bool (optional)
        Flag to set all floating-point arrays in the returned `Rollout` to
        `bfloat16` on accelerator transfer. Halves VRAM usage for rollout buffers with
        negligible effect on training quality. `actions` remain `int32`.
        Default is `True`
    """

    def __init__(
        self,
        envs: EnvSet | EnvGroup,
        *,
        config: RuleTrainerSettings,
        agents_per_env: int = 2,
        max_group_size: int = 8,
        seed: int = 42,
        jit_compile: bool = True,
        cache_dir: str | None = ".cache/jax",
        verbose: bool = True,
        use_bfloat16: bool = True,
    ) -> None:
        _cache_status = cache_status(cache_dir, jit_compile)

        # JAX CPU only guard for bfloat16 - not supported on CPU only
        _cpu_only = all(d.platform == "cpu" for d in jax.devices())
        self.use_bfloat16 = False if _cpu_only else use_bfloat16

        if jit_compile and cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        if isinstance(envs, EnvGroup):
            envs = EnvSet(envs)

        config.verify_params()
        envs.verify_packages()

        # Setup envs
        _unique_env_specs = envs.as_list()
        self._env_specs = _unique_env_specs * agents_per_env
        self._env_specs.sort(key=lambda x: x[0])  # group same-game slots together
        self._unique_env_specs = _unique_env_specs

        self.env_names = [name for name, _ in self._env_specs]
        self.num_envs = len(self._unique_env_specs)
        self.num_trainers = len(self._env_specs)

        self.n_steps = config.n_meta_steps(self.num_trainers)
        self._budget_rng = random.Random(seed)

        # Store params
        self.config = config
        self.agents_per_env = agents_per_env
        self.max_group_size = max_group_size
        self.jit_compile = jit_compile
        self.cache_dir = cache_dir
        self.use_bfloat16 = use_bfloat16

        self._seed = seed
        self._env_groups = envs.groups
        self._cpu = jax.devices("cpu")[0]

        # Init logger - one writer per unique env
        self.logger = MetricsLogger(self.config.logger)
        self.logger.add_writer("meta")

        for name, _ in self._unique_env_specs:
            self.logger.add_writer(f"envs/{name}")

        # Configure RNG keys
        key = jax.random.key(seed)
        self.key, meta_key, *trainer_keys = jax.random.split(key, self.num_trainers + 2)

        # Init meta-agent and optimizer
        self.meta_agent = DiscoAgent(
            config=config.disco_agent,
            key=meta_key,
            jit_compile=self.jit_compile,
        )
        self.meta_optim = optax.adam(config.meta_lr)

        # Init state
        self.state = RuleTrainerState.create(
            self.meta_optim.init(self.meta_agent.get_params()),
            self.num_trainers,
            self.config.batch_size,
            *self.meta_agent.hidden_sizes,
        )
        self.trainers: List[AgentTrainer] = []

        # Checkpointing
        self.cp_manager = CheckpointManager(self.config.checkpoint)
        self.rule_path = Path(self.cp_manager.cp_dir, "final_disco").resolve()

        # Batch gradient functions
        self.max_actions = self._compute_max_actions()
        self._batch_grad_fn: Callable[..., MetaGradOutput] = None  # type: ignore

        self._meta_optim_update: Callable = None  # type: ignore

        # Sebulba actor infrastructure
        # One queue per trainer - actors stay exactly one step ahead of the learner
        # One resume event per trainer - main thread unblocks actor after reset
        self._rollout_queues: List[queue.Queue] = [
            queue.Queue(maxsize=1) for _ in range(self.num_trainers)
        ]
        self._resume_events: List[threading.Event] = [
            threading.Event() for _ in range(self.num_trainers)
        ]
        self._stop_event = threading.Event()
        self._actor_threads: List[threading.Thread] = []

        # Init events so actors start collecting immediately
        for ev in self._resume_events:
            ev.set()

        # init console dashboard
        _n_chunks = math.ceil(self.num_trainers / self.max_group_size)

        if verbose:
            self.console = DiscoConsoleDashboard(
                self.config.console_config(
                    envs=envs.env_categories(),
                    num_trainers=self.num_trainers,
                    n_chunks=_n_chunks,
                    params=self._dummy_params(trainer_keys[0]),
                    complete_path=str(self.rule_path),
                    jit_compile=jit_compile,
                    cache_status=_cache_status,
                )
            )
        else:
            self.console = SimpleDashboard(self.n_steps)

        # Setup - build trainers, warm networks, compile gradient functions
        self._initial_setup(trainer_keys)
        self.console.finish_setup()

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
        envs = make_fn(env_name, self.config.batch_size)

        policy = PolicyAgent(
            envs.single_observation_space,  # type: ignore
            envs.single_action_space,  # type: ignore
            config=config.agent,
            key=rng_key,
            max_actions=self.max_actions,
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

    def _compute_max_actions(self) -> int:
        """
        Probe each unique environment to determine the maximum action count.

        Returns
        -------
        max_actions : int
            Maximum number of discrete actions across all environments
        """
        max_actions = 0

        for env_name, make_fn in self._unique_env_specs:
            env = make_fn(env_name, self.config.batch_size)
            action_space: gym.spaces.Discrete = env.single_action_space  # type: ignore
            max_actions = max(max_actions, action_space.n.item())
            env.close()

        return max_actions

    def _warm_compile(self) -> None:
        """
        Pre-compile all gradient functions before training begins.

        Compiles for every distinct chunk size based on `num_trainers` and `max_group_size`.
        """
        meta_params = self.meta_agent.get_params()

        # Compile meta optimizer update fn (if needed)
        self._meta_optim_update = (
            self.meta_optim.update
            if not self.jit_compile
            else jax.jit(self.meta_optim.update)
        )

        def _stack(x, n):
            """Stack `n` copies of PyTree `x` along a new leading axis."""
            return jax.tree.map(lambda t: jnp.stack([t] * n), x)

        # Use any trainer — all have identical shapes
        trainer = self.trainers[0]

        # Collect real rollouts so shapes match exactly what training uses
        with jax.default_device(self._cpu):
            train_rollout = trainer.collect_stack(self.config.n_updates)
            valid_rollout = trainer.collect_valid()

        p_params, v_params, p_opt, v_opt, adv_ema, td_ema = trainer.get_vmap_inputs()
        disco_h, meta_h = self.state.hidden.get(0)
        action_mask = jnp.arange(self.max_actions) < trainer.n_actions

        # Compile for every distinct chunk size
        chunk_sizes = {min(self.max_group_size, self.num_trainers)}
        remainder = self.num_trainers % self.max_group_size

        if remainder and remainder != min(self.max_group_size, self.num_trainers):
            chunk_sizes.add(remainder)  # Include remainder chunks

        for chunk_size in chunk_sizes:
            _ = self._batch_grad_fn(
                meta_params,
                _stack(p_params, chunk_size),
                _stack(v_params, chunk_size),
                jnp.stack([disco_h] * chunk_size),
                jnp.stack([meta_h] * chunk_size),
                _stack(p_opt, chunk_size),
                _stack(v_opt, chunk_size),
                _stack(adv_ema, chunk_size),
                _stack(td_ema, chunk_size),
                _stack(train_rollout, chunk_size),
                _stack(valid_rollout, chunk_size),
                jnp.stack([action_mask] * chunk_size),
            )

        # Block until compilations are complete
        jax.effects_barrier()

        # Warm per-trainer meta-optim JIT
        dummy_grad = jax.tree.map(jnp.zeros_like, meta_params)
        for t in self.trainers:
            _ = t.meta_optim_update(
                dummy_grad,
                t.meta_opt_state,
                meta_params,
            )
        jax.effects_barrier()

    def _initial_setup(self, trainer_keys: List[chex.PRNGKey]) -> None:
        """
        Builds all agent trainers, initializes meta-optimizers, compiles inference on
        accelerator and CPU, builds vmapped batch-grad function, and pre-compiles
        for all chunk sizes.

        Parameters
        ----------
        trainer_keys : List[chex.PRNGKey]
            List of trainer random number generated keys
        """
        setup_total = 2 * self.num_trainers + 3
        self.console.start_setup(setup_total)

        # Build trainers
        for i, (env_name, make_fn) in enumerate(self._env_specs):
            trainer = AgentTrainer(
                env_name,
                self.config.agent_trainer_config(),
                key=trainer_keys[i],
                logger=self.logger,
                writer_name=f"envs/{env_name}",
                make_fn=make_fn,
                budget_rng=self._budget_rng,
                max_actions=self.max_actions,
                jit_compile=self.jit_compile,
                use_bfloat16=self.use_bfloat16,
            )
            self.trainers.append(trainer)
            self.console.update_setup()

        # Init meta-optimizers and warm one trainer on each device
        warmed_acl = False
        warmed_cpu = False

        for trainer in self.trainers:
            trainer._init_meta_optim(self.meta_agent.get_params())

            if not warmed_acl:
                trainer.warm()
                warmed_acl = True

            with jax.default_device(self._cpu):
                if not warmed_cpu:
                    trainer.warm()
                    _ = trainer.collect_stack(self.config.n_updates)
                    _ = trainer.collect_valid()
                    warmed_cpu = True

            self.console.update_setup()

        # Build single batch-grad function
        self._batch_grad_fn = self._build_batch_grad_fn()
        self.console.update_setup()

        # Pre-compile all chunk sizes so train() starts warm
        self._warm_compile()
        self.console.update_setup()

        # Warm all remaining eager JAX ops that run outside vmapped_fn
        _meta_params = self.meta_agent.get_params()
        _dummy_grad = jax.tree.map(jnp.zeros_like, _meta_params)

        _updates, _ = self._meta_optim_update(
            _dummy_grad,
            self.state.meta_opt_state,
            _meta_params,
        )
        _ = optax.apply_updates(_meta_params, _updates)
        _ = jax.tree.map(lambda g: g / self.num_trainers, _dummy_grad)
        _ = optax.global_norm(_dummy_grad)

        self.console.update_setup()
        jax.effects_barrier()

    def _build_batch_grad_fn(self) -> Callable:
        """
        Build a JIT + `vmap` compiled meta-gradient function.

        `meta_params` is shared across the chunk (`in_axes=None`). All per-trainer
        inputs are batched along the leading chunk dimension.

        Returns
        -------
        fn : Callable
            JIT + `vmap` compiled gradient function
        """
        t = self.trainers[0]

        def _pure_fn(
            meta_p, p_p, v_p, d_h, m_h, p_opt, v_opt, adv_e, td_e, tr, vr, a_mask
        ):
            return self._compute_meta_gradient(
                t,
                meta_p,
                p_p,
                v_p,
                d_h,
                m_h,
                p_opt,
                v_opt,
                adv_e,
                td_e,
                tr,
                vr,
                a_mask,
            )

        return jax.jit(
            jax.vmap(
                _pure_fn,
                in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            )
        )

    def _compute_meta_gradient(
        self,
        trainer: AgentTrainer,
        meta_params: chex.ArrayTree,
        p_params: chex.ArrayTree,
        v_params: chex.ArrayTree,
        disco_h: chex.Array,
        meta_h: chex.Array,
        p_opt_state: chex.ArrayTree,
        v_opt_state: chex.ArrayTree,
        adv_ema: EMAState,
        td_ema: EMAState,
        train_rollouts: Rollout,
        valid_rollout: Rollout,
        action_mask: chex.Array,
    ) -> MetaGradOutput:
        """
        Compute meta-gradient for a single agent through the inner loop.

        Parameters
        ----------
        trainer : AgentTrainer
            Read-only reference (config, ema_utils, functional_forward)
        meta_params : ArrayTree
            Meta-network parameters (shared across a group; `in_axes=None`
            in `vmap`)
        p_params : ArrayTree
            Policy network parameters for this trainer
        v_params : ArrayTree
            Value network parameters for this trainer
        disco_h : chex.Array
            Disco network hidden state `(B, H)`
        meta_h : chex.Array
            Meta-LNN hidden state `(B, H_meta)`
        p_opt_state : ArrayTree
            Policy optimizer state
        v_opt_state : ArrayTree
            Value optimizer state
        adv_ema : EMAState
            Advantage EMA state
        td_ema : EMAState
            TD-error EMA state
        train_rollouts : Rollout
            Training rollouts `(N, B, T, ...)`
        valid_rollout : Rollout
            Validation rollout `(1, T, ...)`
        action_mask : chex.Array
            Boolean mask `(max_actions,)` for valid actions

        Returns
        -------
        out : MetaGradOutput
            Gradient and updated state outputs
        """

        def meta_loss_fn(
            meta_params,
        ) -> Tuple[chex.Array, Tuple[MetaLossAux, EMAState, EMAState]]:
            def _inner_step(
                carry: MetaInnerStepCarry, rollout: Rollout
            ) -> Tuple[MetaInnerStepCarry, Tuple]:
                targets, d_h, m_h = self.meta_agent(
                    rollout,
                    disco_h_state=carry.disco_h,
                    meta_h_state=carry.meta_h,
                    params=meta_params,
                    action_mask=action_mask,
                )

                encoding = rollout.preds.encoding
                actions, discounts = rollout.actions, rollout.discounts

                def inner_policy_loss(p, enc, tgt, act, disc):
                    new_preds = trainer.policy_agent.functional_forward(
                        enc,
                        p,
                        action_mask=action_mask,
                    )
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

                (_, _), p_grads = fwdrev_value_and_grad(
                    inner_policy_loss,
                    has_aux=True,
                )(carry.p_params, encoding, targets, actions, discounts)

                p_updates, new_p_opt = trainer.policy_optim.update(
                    p_grads, carry.p_opt_state, carry.p_params
                )
                new_p_params = optax.apply_updates(carry.p_params, p_updates)

                def value_loss_fn(vp, r, adv, td):
                    value_outs, new_adv_ema, new_td_ema = trainer.compute_value_outs(
                        r,
                        adv,
                        td,
                    )
                    net_out = value_outs.value[:-1]  # type: ignore
                    value_target = jax.lax.stop_gradient(
                        net_out + value_outs.normalized_td
                    )
                    value_loss = (
                        trainer.config.loss_costs.value
                        * 0.5
                        * jnp.square(net_out - value_target).mean()
                    )
                    return value_loss, (value_outs, new_adv_ema, new_td_ema)

                (_, (_, new_adv_ema, new_td_ema)), v_grads = fwdrev_value_and_grad(
                    value_loss_fn, has_aux=True
                )(carry.v_params, rollout, carry.adv_ema, carry.td_ema)

                v_updates, new_v_opt = trainer.value_optim.update(
                    v_grads, carry.v_opt_state, carry.v_params
                )
                new_v_params = optax.apply_updates(carry.v_params, v_updates)

                new_carry = MetaInnerStepCarry(
                    p_params=new_p_params,
                    v_params=new_v_params,
                    disco_h=d_h,
                    meta_h=m_h,
                    p_opt_state=jax.lax.stop_gradient(new_p_opt),
                    v_opt_state=jax.lax.stop_gradient(new_v_opt),
                    adv_ema=new_adv_ema,
                    td_ema=new_td_ema,
                )
                return new_carry, (targets, rollout.target_preds.pi)

            init_carry = MetaInnerStepCarry(
                p_params=p_params,
                v_params=v_params,
                disco_h=disco_h,
                meta_h=meta_h,
                p_opt_state=p_opt_state,
                v_opt_state=v_opt_state,
                adv_ema=adv_ema,
                td_ema=td_ema,
            )

            final_out, (all_targets, all_targets_pi) = jax.lax.scan(
                jax.checkpoint(  # type: ignore
                    _inner_step,
                    policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
                ),
                init_carry,  # type: ignore
                train_rollouts,  # type: ignore
            )

            # Validation pass
            value_outs, _, _ = trainer.compute_value_outs(
                valid_rollout,
                final_out.adv_ema,
                final_out.td_ema,
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
                jax.tree.map(lambda x: x[-1], all_targets),
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

            # Carry final EMA states out of differentiable closure
            final_adv_ema = jax.lax.stop_gradient(final_out.adv_ema)
            final_td_ema = jax.lax.stop_gradient(final_out.td_ema)
            return meta_loss, (aux, final_adv_ema, final_td_ema)

        (meta_loss, (aux, final_adv_ema, final_td_ema)), meta_grad = jax.value_and_grad(
            meta_loss_fn,
            has_aux=True,
        )(meta_params)
        aux: MetaLossAux = aux

        return MetaGradOutput(
            meta_grad=meta_grad,
            disco_h=aux.disco_h,
            meta_h=aux.meta_h,
            p_params=aux.p_params,
            v_params=aux.v_params,
            v_opt_state=aux.v_opt_state,
            pg_loss=aux.pg_loss,
            entropy_loss=aux.entropy_loss,
            reg_loss=aux.reg_loss,
            meta_loss=meta_loss,
            advantages=aux.value_outs.advantages,
            normalized_advantages=aux.value_outs.normalized_advantages,
            adv_ema=final_adv_ema,
            td_ema=final_td_ema,
        )

    def _actor_loop(self, trainer_idx: int) -> None:
        """
        Persistent actor thread for a single trainer.

        Runs continuously until `_stop_event` is set. On each iteration:

            1. Wait for `_resume_events[trainer_idx]` - cleared by main thread
               during a reset and re-set once the reset is complete.
            2. Check `needs_reset` - if `True`, push `_RESET_SENTINEL` onto
               the queue and wait for the main thread to handle the reset.
            3. Otherwise, collect `train` + `valid` rollouts and push them as a tuple.

        The queue has `maxsize=1`, so this thread blocks automatically when the
        trainer hasn't yet consumed the previous rollout, keeping actors exactly
        one meta-step ahead.

        Parameters
        ----------
        trainer_idx : int
            Trainer index for `self.trainers` and `self._rollout_queues`
        """
        q = self._rollout_queues[trainer_idx]
        resume = self._resume_events[trainer_idx]

        while not self._stop_event.is_set():
            # Wait for main thread to handle a reset
            resume.wait()

            if self._stop_event.is_set():
                break

            trainer = self.trainers[trainer_idx]

            # Signal main thread to reset
            if trainer.needs_reset:
                resume.clear()
                q.put(_RESET_SENTINEL)
                continue

            # Force rollout threading on CPU
            with jax.default_device(self._cpu):
                train_rollout = trainer.collect_stack(self.config.n_updates)
                valid_rollout = trainer.collect_valid()

            q.put((train_rollout, valid_rollout))

    def _start_actors(self) -> None:
        """
        Spawn one persistent daemon actor thread per trainer.

        Called once at the start of `train()`. Daemon threads are
        automatically killed if the main process exists.
        """
        self._stop_event.clear()
        self._actor_threads = []

        for i in range(self.num_trainers):
            t = threading.Thread(
                target=self._actor_loop,
                args=(i,),
                name=f"actor-{i}",
                daemon=True,
            )
            t.start()
            self._actor_threads.append(t)

    def _stop_actors(self) -> None:
        """
        Signal all actor threads to stop running, waitings for thread completion
        and then terminates them.
        """
        # Signal all actors to stop looping
        self._stop_event.set()

        # Unblock any threads waiting on resume or a full queue
        for ev in self._resume_events:
            ev.set()

        for q in self._rollout_queues:
            try:
                q.get_nowait()
            except queue.Empty:
                pass

        # Wait for each thread to naturally terminate
        for t in self._actor_threads:
            t.join(timeout=1)

    def _get_rollouts(self, indices: List[int]) -> Tuple[List[Rollout], List[Rollout]]:
        """
        Pull one rollout pair per trainer from the actor queues.

        Blocks until each trainer's queue has an item. Handles reset
        sentinels inline — calls `reset_trainer` on the main thread
        then sets the actor's resume event so it can continue collecting.

        Parameters
        ----------
        indices : List[int]
            Trainer indices to collect rollouts for

        Returns
        -------
        train_rollouts : List[Rollout]
            Training rollouts in group-index order
        valid_rollouts : List[Rollout]
            Validation rollouts in group-index order
        """
        train_rollouts: List[Rollout] = []
        valid_rollouts: List[Rollout] = []

        for idx in indices:
            q = self._rollout_queues[idx]
            resume = self._resume_events[idx]

            while True:
                item = q.get()

                if item is _RESET_SENTINEL:
                    # Actor is blocked on resume - safe to mutate trainer slot
                    self.reset_trainer(idx)
                    resume.set()  # Unblock actor to collect fresh rollout
                    continue  # Loop to get the rollout

                tr, vr = item
                train_rollouts.append(tr)
                valid_rollouts.append(vr)
                break

        return train_rollouts, valid_rollouts

    def _process_chunk(
        self,
        chunk_indices: List[int],
        train_rollouts: List[Rollout],
        valid_rollouts: List[Rollout],
        meta_params: chex.ArrayTree,
        accumulated_grad: chex.ArrayTree,
        stats: MetaStepStats,
    ) -> chex.ArrayTree:
        """
        Compute meta-gradients for a chunk of trainers.

        Rollouts are padded from `n_actions` to `max_actions` before stacking,
        and an action mask is threaded through for masked operations.

        Parameters
        ----------
        chunk_indices : List[int]
            Trainer indices in this chunk
        train_rollouts : List[Rollout]
            Pre-collected training rollouts, one per trainer in chunk order
        valid_rollouts : List[Rollout]
            Pre-collected validation rollouts, one per trainer in chunk order
        meta_params : chex.ArrayTree
            Current shared meta-network parameters
        accumulated_grad : chex.ArrayTree
            Running gradient accumulator to add into
        stats : MetaStepStats
            Step statistics accumulator

        Returns
        -------
        accumulated_grad : chex.ArrayTree
            Updated gradient accumulator
        """
        vmapped_fn = self._batch_grad_fn
        chunk_trainers = [self.trainers[i] for i in chunk_indices]

        # Stack chunk inputs along a leading group dimension for vmap
        stacked_train = stack_pytrees(train_rollouts)
        stacked_valid = stack_pytrees(valid_rollouts)

        (
            stacked_p_params,
            stacked_v_params,
            stacked_p_opt,
            stacked_v_opt,
            stacked_adv_ema,
            stacked_td_ema,
        ) = [
            stack_pytrees(field)  # type: ignore
            for field in zip(*[t.get_vmap_inputs() for t in chunk_trainers])
        ]

        stacked_disco_h, stacked_meta_h = [
            jnp.stack(hs)
            for hs in zip(*[self.state.hidden.get(i) for i in chunk_indices])
        ]

        # Build per-trainer action masks: (chunk, max_actions)
        stacked_action_masks = jnp.stack(
            [jnp.arange(self.max_actions) < t.n_actions for t in chunk_trainers]
        )

        # Single accelerator call for entire chunk
        chunk_out: MetaGradOutput = vmapped_fn(
            meta_params,
            stacked_p_params,
            stacked_v_params,
            stacked_disco_h,
            stacked_meta_h,
            stacked_p_opt,
            stacked_v_opt,
            stacked_adv_ema,
            stacked_td_ema,
            stacked_train,
            stacked_valid,
            stacked_action_masks,
        )

        # Post-batch: apply updates and accumulate gradients
        for i, (idx, trainer) in enumerate(zip(chunk_indices, chunk_trainers)):
            p_params_i = unstack_pytree(chunk_out.p_params, i)
            v_params_i = unstack_pytree(chunk_out.v_params, i)
            v_opt_i = unstack_pytree(chunk_out.v_opt_state, i)
            meta_grad_i = unstack_pytree(chunk_out.meta_grad, i)

            trainer.policy_agent.update_params(p_params_i)  # type: ignore
            trainer.target_agent.update_params(p_params_i)  # type: ignore
            trainer.value_agent.update_params(v_params_i)  # type: ignore
            trainer.state = trainer.state.update_value_opt(v_opt_i)

            # Write updated EMA states back to trainer for next meta-step
            adv_ema_i: EMAState = unstack_pytree(chunk_out.adv_ema, i)  # type: ignore
            td_ema_i: EMAState = unstack_pytree(chunk_out.td_ema, i)  # type: ignore
            trainer.state = trainer.state.update_ema(adv_ema_i, td_ema_i)

            self.state = self.state.update_hidden(
                idx,
                chunk_out.disco_h[i],  # type: ignore
                chunk_out.meta_h[i],  # type: ignore
            )

            # Norm gradient magnitude via per trainer optim before accumulating
            normed_grad, new_meta_opt_state = trainer.meta_optim_update(
                meta_grad_i,
                trainer.meta_opt_state,
                meta_params,
            )
            trainer.meta_opt_state = new_meta_opt_state

            accumulated_grad = jax.tree.map(
                lambda acc, g: acc + g,
                accumulated_grad,
                normed_grad,
            )

            # Log metrics
            self._log_meta_metrics(
                idx,
                chunk_out.pg_loss[i],  # type: ignore
                chunk_out.entropy_loss[i],  # type: ignore
                chunk_out.reg_loss[i],  # type: ignore
                chunk_out.meta_loss[i],  # type: ignore
                chunk_out.advantages[i],  # type: ignore
                chunk_out.normalized_advantages[i],  # type: ignore
            )

            losses_i = LossStatistics(
                meta=chunk_out.meta_loss[i],  # type: ignore
                policy_gradient=chunk_out.pg_loss[i],  # type: ignore
                entropy=chunk_out.entropy_loss[i],  # type: ignore
                regularization=chunk_out.reg_loss[i],  # type: ignore
            )
            stats.record(trainer.ep_return(), trainer.ep_length(), losses_i)

        return accumulated_grad

    def train(self) -> None:
        """
        Performs meta-training loop to discover an RL update rule.
        Uses Sebulba-style decoupled actor/learner pipeline.

        One persistent daemon thread per trainer (actor) continuously collects
        rollouts and pushes them onto a depth-1 queue. The main thread (learner)
        drains those queues group-by-group, runs accelerator gradient computation
        for each group, then applies the averaged meta-update.

        All actor threads run concurrently, so CPU environment stepping for every
        trainer overlaps with accelerator gradient computations.

        Includes
        --------
        1. For each meta-step:

            a. Iterate through all agent trainers sequentially
            b. For each trainer, collect rollouts and perform inner loop updates
            c. Accumulate meta-gradients through the learning process
            d. Average gradients across all environments and update meta-network

        2. Log metrics and checkpoints periodically
        """
        self.console.start_training()
        self._start_actors()

        try:
            for step in range(self.state.meta_step, self.n_steps):
                accumulated_grad = jax.tree.map(
                    jnp.zeros_like,
                    self.meta_agent.get_params(),
                )
                stats = MetaStepStats(self.num_trainers)
                meta_params = self.meta_agent.get_params()

                # Flat chunking over all trainers
                all_indices = list(range(self.num_trainers))
                train_rollouts, valid_rollouts = self._get_rollouts(all_indices)

                for start in range(0, self.num_trainers, self.max_group_size):
                    end = min(start + self.max_group_size, self.num_trainers)
                    chunk_indices = all_indices[start:end]
                    chunk_train = train_rollouts[start:end]
                    chunk_valid = valid_rollouts[start:end]

                    chunk_envs = [self.trainers[i].env_name for i in chunk_indices]
                    self.console.update_progress(
                        "Inner Updates",
                        chunk_size=len(chunk_indices),
                        env_names=chunk_envs,
                    )

                    # Compute gradients for this agent
                    accumulated_grad = self._process_chunk(
                        chunk_indices,
                        chunk_train,
                        chunk_valid,
                        meta_params,
                        accumulated_grad,
                        stats,
                    )

                # Apply average gradients through single shared optimizer
                avg_grad = jax.tree.map(
                    lambda g: g / self.num_trainers,
                    accumulated_grad,
                )
                self._apply_meta_update(avg_grad)

                # Log metrics
                grad_norm = float(optax.global_norm(avg_grad))
                metrics = {
                    "meta/grad_norm": grad_norm,
                    **stats.summary("meta/"),
                    "meta/step": step,
                }
                self.logger.log("meta", step, metrics)

                self.console.update_stats(**stats.rewards_as_dict())
                self.console.update_losses(
                    **stats.losses_as_dict(),
                    gradient_norm=grad_norm,
                )

                # Checkpoint periodically
                if step % self.config.checkpoint.freq == 0:
                    self.save_checkpoint()

                self.console.update_progress("Meta Steps")

        except (KeyboardInterrupt, SystemExit):
            jax.debug.print("Terminating.")
            exit()
        finally:
            self._stop_actors()

        # Final cleanup
        self.save_checkpoint(force=True)
        self.save_rule()

        self.close()
        self.console.finish_training()

    @classmethod
    def restore(
        cls,
        checkpoint_dir: str,
        *,
        additional_steps: int | None = None,
        jit_compile: bool = True,
        cache_dir: str | None = ".cache/jax",
        verbose: bool = True,
    ) -> Self:
        """
        Restore meta-training state from the latest checkpoint.

        Loads meta-agent parameters, optimizer state, and hidden states,
        then resets all agent trainers.

        Call `train()` after to continue training from the restored `meta_step`.

        Parameters
        ----------
        checkpoint_dir : str
            Path to the checkpoint directory containing `meta_run.json`
            and the Orbax checkpoint subdirectories
        additional_steps : int (optional)
            Number of additional meta-training steps beyond the restored
            `meta_step`. When `None`, continues to the original
            `n_meta_steps` target from config. Default is `None`
        jit_compile : bool (optional)
            Flag to enable/disable JIT compilation. Default is `True`
        cache_dir : str | None (optional)
            Directory path for JAX's persistent XLA compilation cache.
            Set to `None` to disable. Default is `".cache/jax"`
        verbose : bool (optional)
            Dashboard display settings. Default is `True`.
                - If `True` - renders the full Rich dashboard
                - If `False` - uses a lightweight `tqdm` progress bar instead

        Returns
        -------
        trainer : RuleTrainer
            A restored `RuleTrainer` ready to use normally

        Raises
        ------
        file_error : FileNotFoundError
            If `meta_run.json` is not found in `checkpoint_dir`
        checkpoint_error : ValueError
            If no checkpoint exists in `checkpoint_dir`
        """
        cp_dir = Path(checkpoint_dir)
        meta_run_path = cp_dir / "meta_run.json"

        if not meta_run_path.exists():
            raise FileNotFoundError(
                f"No 'meta_run.json' found in '{checkpoint_dir}'."
                "Ensure the checkpoint was saved by 'RuleTrainer'."
            )

        meta_run = json.loads(meta_run_path.read_text())

        # Reconstruct environment groups from saved serialized data
        envs = EnvSet(*[EnvGroup.load(g) for g in meta_run["envs"]])
        config = load_config(RuleTrainerSettings, meta_run["config"])

        # Construct trainer
        trainer = cls(
            envs,
            config=config,
            agents_per_env=meta_run["agents_per_env"],
            max_group_size=meta_run["max_group_size"],
            seed=meta_run["seed"],
            jit_compile=jit_compile,
            cache_dir=cache_dir,
            verbose=verbose,
        )

        # Restore meta-agent params and RuleTrainerState (includes meta_step)
        if not trainer.load_checkpoint():
            raise ValueError(
                f"No checkpoint found in '{checkpoint_dir}'. "
                "Ensure the directory contains valid orbax checkpoint files."
            )

        # Extend training if additional steps requested
        if additional_steps is not None:
            trainer.n_steps = trainer.state.meta_step + additional_steps

        return trainer

    def _log_meta_metrics(
        self,
        trainer_idx: int,
        pg_loss: chex.Array,
        entropy_loss: chex.Array,
        reg_loss: chex.Array,
        total_loss: chex.Array,
        advantages: chex.Array,
        normalized_advantages: chex.Array,
    ) -> None:
        """
        Log meta-training metrics for a single trainer.

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
        advantages : chex.Array
            Raw advantages from validation rollout
        normalized_advantages : chex.Array
            EMA-normalised advantages from the validation rollout
        """
        metrics = {
            "meta/pg_loss": float(pg_loss),
            "meta/entropy_loss": float(entropy_loss),
            "meta/reg_loss": float(reg_loss),
            "meta/total_loss": float(total_loss),
            "meta/advantages": float(jnp.mean(advantages)),
            "meta/normalized_advantages": float(jnp.mean(normalized_advantages)),
        }
        self.logger.log(
            f"envs/{self.env_names[trainer_idx]}",
            self.state.meta_step,
            metrics,
        )

    def _apply_meta_update(self, avg_grad: chex.ArrayTree) -> None:
        """
        Apply the averaged meta-gradient through the shared meta-optimizer.

        Parameters
        ----------
        avg_grad : chex.ArrayTree
            Averaged gradients across all agents
        """
        meta_params = self.meta_agent.get_params()

        updates, new_opt_state = self._meta_optim_update(
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
            - `meta_run.json` - config and environment set

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

        meta_run = {
            "config": dump_config(self.config),
            "agents_per_env": self.agents_per_env,
            "max_group_size": self.max_group_size,
            "seed": self._seed,
            "envs": [g.dump() for g in self._env_groups],
        }

        meta_run_path = Path(self.cp_manager.cp_dir, "meta_run.json")
        meta_run_path.write_text(json.dumps(meta_run, indent=2))

        return True

    def load_checkpoint(self, step: int | None = None) -> bool:
        """
        Restore training state from a checkpoint.

        All agent trainers are reset to a fresh state after restoring the
        meta-agent parameters and optimizer state.

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
        Reset a trainer to its initial state - fresh agent parameters,
        fresh optimizer state, zeroed hidden states, new step budget.

        Called by the training loop when `trainer.needs_reset=True`.
        The new trainer draws a fresh budget from the lifetime distribution,
        so successive lifetimes of the same slot are independently sampled.

        Parameters
        ----------
        trainer_idx : int
            Index of the trainer to reset
        """
        env_name, make_fn = self._env_specs[trainer_idx]
        self.trainers[trainer_idx].close()

        # Fresh RNG keys
        self.key, new_key = jax.random.split(self.key, 2)

        # Init new trainer
        new_trainer = AgentTrainer(
            env_name,
            self.config.agent_trainer_config(),
            key=new_key,
            logger=self.logger,
            writer_name=f"envs/{env_name}",
            make_fn=make_fn,
            budget_rng=self._budget_rng,
            max_actions=self.max_actions,
            jit_compile=self.jit_compile,
            use_bfloat16=self.use_bfloat16,
        )

        # Init fresh per-trainer meta-gradient optimizer
        new_trainer._init_meta_optim(self.meta_agent.get_params())
        self.trainers[trainer_idx] = new_trainer

        # Reset hidden state for this trainer
        self.state: RuleTrainerState = self.state.update_hidden(trainer_idx, None, None)

        self.logger.log(
            f"envs/{env_name}",
            self.state.meta_step,
            {"lifetime/reset": 1, "lifetime/budget": new_trainer._step_budget},
        )

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
