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

import queue
import random
import shutil
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from velora.cli.base.simple import SimpleDashboard
from velora.cli.disco.dashboard import DiscoConsoleDashboard
from velora.cli.disco.settings import DiscoParamsSettings
from velora.disco.agent import DiscoAgent, DiscoValueAgent, PolicyAgent
from velora.disco.config.settings import AgentTrainerSettings, RuleTrainerSettings
from velora.disco.config.state import AgentTrainerState, RuleTrainerState
from velora.disco.ema import EMAState, MovingAverage
from velora.disco.inputs import ActionGroup, MetaLossFnInputs
from velora.disco.outputs import (
    AgentLossAux,
    AgentLosses,
    DiscoAgentOutput,
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
        jit_compile: bool = False,
        use_bfloat16: bool = True,
    ) -> None:
        self.config = config
        self.env_name = env_name

        self.envs = make_fn(self.env_name, 1)  # B=1

        self.logger = logger
        self.writer_name = writer_name
        self.jit_compile = jit_compile
        self.use_bfloat16 = use_bfloat16

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
        )  # (1, H)

        self.meta_optim: optax.GradientTransformation = None  # type: ignore
        self.meta_opt_state: optax.OptState = None  # type: ignore

        # Cached JIT-compiled meta-gradient function (built once on first use)
        self._grad_fn: Callable = None  # type: ignore

        # Lifetime tracking
        self._env_steps: int = 0
        self._step_budget: int = sample_budget(budget_rng)

        # Episode tracking
        self.episode_tracker = EpisodeTracker(1)
        self._collection_step = 0

        # Pre-allocated buffers
        self.train_buffer = RolloutBuffer(
            n_rollouts=self.config.n_updates,
            n_envs=1,
            seq_len=self.config.seq_len,
            n_actions=self.n_actions,
            encoding_dim=self.policy_agent.encoding_dim,
            prediction_dim=self.config.agent.prediction_size,
            q_dim=self.config.agent.q_size,
            use_bfloat16=self.use_bfloat16,
        )
        self.valid_buffer = RolloutBuffer(
            n_rollouts=1,
            n_envs=1,
            seq_len=self.config.seq_len * 2,
            n_actions=self.n_actions,
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

    def warm(self) -> None:
        """
        Performs an initial forward pass through the agent networks.

        Includes -
            1. Materializing parameters before optimizer init
            2. JIT compile caching
        """
        dummy_obs = jnp.zeros(
            (1, *self.obs_space.shape),
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
            hidden = hidden.reset_on_done(discounts, self.n_actions)

            obs = next_obs

        # Update trainer state
        self.state = self.state.update_hidden(hidden)
        self.state = self.state.update_obs(jnp.asarray(obs))

    def collect_stack(self, n_rollouts: int) -> Rollout:
        """
        Collect a stack of `N` rollout training trajectories.

        Writes each rollout directly into `train_buffer` and returns a single
        stacked `Rollout` with shape `(N, 1, T, ...)`.

        Parameters
        ----------
        n_rollouts : int
            Number of rollouts to collect. Must match the `n_rollouts`
            dimension in the `train_buffer`

        Returns
        -------
        stack : Rollout
            Stacked rollouts with shape `(N, 1, T, ...)`.
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
        Maximum number of trainers to `vmap` simultaneously. Reduce if accelerator
        is out of memory (OOM). Default is `8`
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
            1,
            *self.meta_agent.hidden_sizes,
        )
        self.trainers: List[AgentTrainer] = []

        # Checkpointing
        self.cp_manager = CheckpointManager(self.config.checkpoint)
        self.rule_path = Path(self.cp_manager.cp_dir, "final_disco").resolve()

        # Per-action group batch gradient functions
        self.action_groups: List[ActionGroup] = []
        self._batch_grad_fns: Dict[int, Callable[..., MetaGradOutput]] = {}

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
        if verbose:
            self.console = DiscoConsoleDashboard(
                self.config.console_config(
                    envs=envs.env_categories(),
                    num_trainers=self.num_trainers,
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
        envs = make_fn(env_name, 1)

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

    def _count_batch_groups(self) -> int:
        """
        Computes the number of action groups that need batch-grad compilation.

        Uses unique environments only - duplicate trainer slots for the same
        game share an action space and don't require additional compilations.

        Returns
        -------
        count : int
            Action group count for batch-grad compilation
        """
        buckets: Dict[int, int] = defaultdict(int)

        for env_name, make_fn in self._unique_env_specs:
            env = make_fn(env_name, 1)
            action_space: gym.spaces.Discrete = env.single_action_space  # type: ignore
            buckets[action_space.n.item()] += self.agents_per_env
            env.close()

        return sum(1 for count in buckets.values() if count > 1)

    def _build_action_groups(self) -> List[ActionGroup]:
        """
        Groups trainers by `n_actions` for vmap batching.

        Groups are sorted largest-first so the most impactful `vmap` runs
        are first each step.

        Returns
        -------
        groups : List[ActionGroup]
            A list of grouped action trainers
        """
        buckets: Dict[int, List[int]] = defaultdict(list)

        for idx, trainer in enumerate(self.trainers):
            buckets[trainer.n_actions].append(idx)

        groups = [ActionGroup(n_actions=n, indices=idxs) for n, idxs in buckets.items()]
        groups.sort(key=lambda g: (-len(g.indices), g.n_actions))
        return groups

    def _initial_setup(self, trainer_keys: List[chex.PRNGKey]) -> None:
        """
        Build all trainers, warm networks, compile gradient functions, and
        pre-compile batch-grad functions for each action group.

        Parameters
        ----------
        trainer_keys : List[chex.PRNGKey]
            List of trainer random number generated keys
        """
        n_batch_groups = self._count_batch_groups()
        setup_total = 2 * self.num_trainers + n_batch_groups
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
                jit_compile=self.jit_compile,
                use_bfloat16=self.use_bfloat16,
            )
            self.trainers.append(trainer)
            self.console.update_setup()

        # Warm networks and compile per-trainer grad functions
        seen_shapes: set[int] = set()

        for trainer_idx, trainer in enumerate(self.trainers):
            trainer._init_meta_optim(self.meta_agent.get_params())

            if trainer.n_actions not in seen_shapes:
                dummy_train = trainer.train_buffer.to_rollout_zeros()
                dummy_valid = trainer.valid_buffer.to_rollout_zeros()[0]

                # Trigger JIT compile for inference networks
                _ = self.meta_agent(dummy_train[0])
                trainer.warm()

                # Build and cache grad fn; triggering compile with dummy data
                self._build_trainer_grad_fn(trainer)
                disco_h, meta_h = self.state.hidden.get(trainer_idx)
                dummy_inputs = MetaLossFnInputs(
                    policy_params=trainer.policy_agent.get_params(),
                    value_params=trainer.value_agent.get_params(),
                    disco_h=disco_h,
                    meta_h=meta_h,
                    p_opt_state=trainer.state.policy_opt_state,
                    v_opt_state=trainer.state.value_opt_state,
                    adv_ema=trainer.state.adv_ema,
                    td_ema=trainer.state.td_ema,
                    train_rollouts=dummy_train,
                    valid_rollout=dummy_valid,
                )
                _ = trainer._grad_fn(self.meta_agent.get_params(), dummy_inputs)
                seen_shapes.add(trainer.n_actions)
            else:
                trainer.warm()
                self._build_trainer_grad_fn(trainer)

            self.console.update_setup()

        # Build action groups and compile batch-grad functions
        self.action_groups = self._build_action_groups()

        for group in self.action_groups:
            fn = self._build_batch_grad_fn(group)

            # Warm-up with a size-1 dummy call to trigger XLA compile
            t = self.trainers[group.indices[0]]
            dummy_train = t.train_buffer.to_rollout_zeros()
            dummy_valid = t.valid_buffer.to_rollout_zeros()[0]
            disco_h, meta_h = self.state.hidden.get(group.indices[0])

            _ = fn(
                self.meta_agent.get_params(),
                jax.tree.map(lambda x: x[None], t.policy_agent.get_params()),
                jax.tree.map(lambda x: x[None], t.value_agent.get_params()),
                disco_h[None],  # type: ignore
                meta_h[None],  # type: ignore
                jax.tree.map(lambda x: x[None], t.state.policy_opt_state),
                jax.tree.map(lambda x: x[None], t.state.value_opt_state),
                jax.tree.map(lambda x: x[None], t.state.adv_ema),
                jax.tree.map(lambda x: x[None], t.state.td_ema),
                jax.tree.map(lambda x: x[None], dummy_train),
                jax.tree.map(lambda x: x[None], dummy_valid),
            )

            self._batch_grad_fns[group.n_actions] = fn
            self.console.update_setup()

    def _build_batch_grad_fn(self, group: ActionGroup) -> Callable:
        """
        Build a JIT + `vmap` compiled meta-gradient function for a trainer group.

        `meta_params` is shared across the group (`in_axes=None`). All per-trainer
        inputs are batched along the leading group dimension.

        Parameters
        ----------
        group : ActionGroup
            The trainer group to build the function for

        Returns
        -------
        fn : Callable
            JIT + `vmap` compiled gradient functions
        """
        t = self.trainers[group.indices[0]]

        def _pure_fn(meta_p, p_p, v_p, d_h, m_h, p_opt, v_opt, adv_e, td_e, tr, vr):
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
            )

        return jax.jit(jax.vmap(_pure_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

    def _build_trainer_grad_fn(self, trainer: AgentTrainer) -> None:
        """
        Build and cache the JIT-compiled meta-gradient function for a trainer.
        Stored on `trainer._grad_fn`.

        Parameters
        ----------
        trainer : AgentTrainer
            The agent trainer to build the gradient function for
        """

        def meta_loss_fn(
            meta_params,
            inputs: MetaLossFnInputs,
        ) -> Tuple[chex.Array, MetaLossAux]:
            """
            Meta-loss function.

            Runs inner loop with MixFlow-MG reparameterization, then
            computes policy gradient on validation data.

            Parameters
            ----------
            meta_params : optax.Params
                Meta agent parameters (differentiated)
            inputs : MetaLossFnInputs
                All non-differentiated JAX inputs for this step

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

                (_, _), p_grads = fwdrev_value_and_grad(
                    inner_policy_loss,
                    has_aux=True,
                )(carry.p_params, encoding, targets, actions, discounts)

                # Apply inner update
                p_updates, new_p_opt = trainer.policy_optim.update(
                    p_grads,
                    carry.p_opt_state,
                    carry.p_params,
                )
                new_p_params = optax.apply_updates(carry.p_params, p_updates)

                # Value update
                _, new_adv_ema, new_td_ema = trainer.compute_value_outs(
                    rollout,
                    carry.adv_ema,
                    carry.td_ema,
                )
                _, new_v_params, new_v_opt = trainer.update_value(
                    rollout,
                    value_params=carry.v_params,
                    value_opt_state=carry.v_opt_state,
                    adv_ema=carry.adv_ema,
                    td_ema=carry.td_ema,
                )

                new_carry = MetaInnerStepCarry(
                    p_params=new_p_params,
                    v_params=new_v_params,
                    disco_h=d_h,
                    meta_h=m_h,
                    p_opt_state=new_p_opt,
                    v_opt_state=new_v_opt,
                    adv_ema=new_adv_ema,
                    td_ema=new_td_ema,
                )
                return new_carry, (targets, rollout.target_preds.pi)

            # Run inner loop
            init_carry = MetaInnerStepCarry(
                p_params=inputs.policy_params,
                v_params=inputs.value_params,
                disco_h=inputs.disco_h,
                meta_h=inputs.meta_h,
                p_opt_state=inputs.p_opt_state,
                v_opt_state=inputs.v_opt_state,
                adv_ema=inputs.adv_ema,
                td_ema=inputs.td_ema,
            )

            final_out, (all_targets, all_targets_pi) = jax.lax.scan(
                jax.checkpoint(_inner_step),  # type: ignore
                init_carry,
                inputs.train_rollouts,
            )

            # Compute value outputs on validation rollout
            value_outs, _, _ = trainer.compute_value_outs(
                inputs.valid_rollout,
                final_out.adv_ema,
                final_out.td_ema,
            )
            adv = jax.lax.stop_gradient(value_outs.normalized_advantages)

            pg_loss = compute_policy_gradient_loss(
                inputs.valid_rollout.preds.pi,
                inputs.valid_rollout.actions,
                adv,
            ).mean()
            entropy_loss = compute_entropy_loss(
                inputs.valid_rollout.preds.pi,
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

        trainer._grad_fn = jax.jit(
            jax.value_and_grad(meta_loss_fn, argnums=0, has_aux=True)
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
            Disco network hidden state `(1, H)`
        meta_h : chex.Array
            Meta-LNN hidden state `(1, H_meta)`
        p_opt_state : ArrayTree
            Policy optimizer state.
        v_opt_state : ArrayTree
            Value optimizer state.
        adv_ema : EMAState
            Advantage EMA state.
        td_ema : EMAState
            TD-error EMA state.
        train_rollouts : Rollout
            Training rollouts `(N, 1, T, ...)`
        valid_rollout : Rollout
            Validation rollout `(1, T, ...)`

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
                )

                encoding = rollout.preds.encoding
                actions, discounts = rollout.actions, rollout.discounts

                def inner_policy_loss(p, enc, tgt, act, disc):
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
                    p_opt_state=new_p_opt,
                    v_opt_state=new_v_opt,
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
                jax.checkpoint(_inner_step),  # type: ignore
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

    def _get_rollouts(self, group: ActionGroup) -> Tuple[List[Rollout], List[Rollout]]:
        """
        Pull one rollout pair per trainer from the actor queues for all
        trainers in `group`.

        Blocks until each trainer's queue has an item. Handles reset
        sentinels inline — calls `reset_trainer` on the main thread
        then sets the actor's resume event so it can continue collecting.

        Parameters
        ----------
        group : ActionGroup
            The trainer group whose rollouts to collect

        Returns
        -------
        train_rollouts : List[Rollout]
            Training rollouts in group-index order
        valid_rollouts : List[Rollout]
            Validation rollouts in group-index order
        """
        train_rollouts: List[Rollout] = []
        valid_rollouts: List[Rollout] = []

        for idx in group.indices:
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

    def _process_group_batch(
        self,
        group: ActionGroup,
        train_rollouts: List[Rollout],
        valid_rollouts: List[Rollout],
        meta_params: chex.ArrayTree,
        accumulated_grad: chex.ArrayTree,
        stats: MetaStepStats,
    ) -> chex.ArrayTree:
        """
        Compute meta-gradients for all trainers in a group.

        All trainers in the group (including singletons) are processed via
        the vmapped function in chunks of at most `max_group_size`.

        Parameters
        ----------
        group : ActionGroup
            The trainer group to process
        train_rollouts : List[Rollout]
            Pre-collected training rollouts, one per trainer in group order
        valid_rollouts : List[Rollout]
            Pre-collected validation rollouts, one per trainer in group order
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
        vmapped_fn = self._batch_grad_fns[group.n_actions]
        chunk_starts = range(0, len(group.indices), self.max_group_size)

        for start in chunk_starts:
            end = start + self.max_group_size
            chunk_idx = group.indices[start:end]

            chunk_trainers = [self.trainers[i] for i in chunk_idx]
            chunk_train = train_rollouts[start:end]
            chunk_valid = valid_rollouts[start:end]

            # Stack chunk inputs along a leading group dimension for vmap
            stacked_train = stack_pytrees(chunk_train)
            stacked_valid = stack_pytrees(chunk_valid)

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
                for hs in zip(*[self.state.hidden.get(i) for i in chunk_idx])
            ]

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
            )

            # Post-batch: apply updates and accumulate gradients
            for i, (idx, trainer) in enumerate(zip(chunk_idx, chunk_trainers)):
                self.console.update_progress(
                    "Inner Updates",
                    env_name=trainer.env_name,
                )

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
                normed_grad, new_meta_opt_state = trainer.meta_optim.update(
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

        Includes -
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
            for step in range(self.n_steps):
                accumulated_grad = jax.tree.map(
                    jnp.zeros_like,
                    self.meta_agent.get_params(),
                )
                stats = MetaStepStats(self.num_trainers)
                meta_params = self.meta_agent.get_params()

                # Iterate through each action group
                for group in self.action_groups:
                    train_rollouts, valid_rollouts = self._get_rollouts(group)

                    # Compute gradients for this agent
                    accumulated_grad = self._process_group_batch(
                        group,
                        train_rollouts,
                        valid_rollouts,
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

        except KeyboardInterrupt:
            self._stop_actors()
            self.close()
            self.console.finish_training()
            return
        finally:
            self._stop_actors()

        # Final cleanup
        self.save_checkpoint(force=True)
        self.save_rule()

        self.close()
        self.console.finish_training()

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
            jit_compile=self.jit_compile,
            use_bfloat16=self.use_bfloat16,
        )

        # Rebuild grad_fn closure
        self._build_trainer_grad_fn(new_trainer)

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
