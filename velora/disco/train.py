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

import math
import random
from typing import Callable, Dict, List, Self, Tuple

import chex
import envrax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from velora.cli.disco.dashboard import DiscoConsoleDashboard
from velora.cli.disco.settings import DiscoParamsSettings
from velora.cli.disco.simple import SimpleDashboard
from velora.compute.loss import (
    compute_gaussian_entropy_loss,
    compute_gaussian_policy_gradient_loss,
)
from velora.compute.mixflow import fwdrev_value_and_grad
from velora.disco.agent import (
    DiscoAgent,
    DiscoValueAgent,
    PolicyAgent,
)
from velora.disco.config.metadata import RuleTrainerMetadata
from velora.disco.config.settings import AgentTrainerSettings, RuleTrainerSettings
from velora.disco.config.state import (
    AgentTrainerHiddenStates,
    AgentTrainerState,
    RuleTrainerState,
)
from velora.disco.ema import EMAState, MovingAverage
from velora.disco.outputs import (
    ChunkLogData,
    DiscoAgentOutput,
    HiddenShapeCache,
    LossStatistics,
    MetaGradOutput,
    MetaInnerStepCarry,
    MetaLossAux,
    MetaStepStats,
    PolicyAgentOutput,
    ValueOutputs,
)
from velora.disco.pool import TrainerPool
from velora.disco.rollouts import Rollout
from velora.disco.utils.budget import sample_budget
from velora.disco.utils.compute import compute_value_outputs
from velora.disco.utils.loss import (
    compute_meta_reg_loss,
    compute_policy_loss,
)
from velora.disco.warm import WarmCompile
from velora.nn.optim import scale_by_adan_no_denom
from velora.tracking.episode import EpisodeTracker
from velora.tracking.logger import MetricsLogger, RuntimeLogger
from velora.tracking.manager import CheckpointManager
from velora.tracking.settings import RunSettings
from velora.utils.config import dump_config, load_config
from velora.utils.format import cache_status
from velora.utils.seed import get_rng_key_data
from velora.utils.transforms import squeeze_time


def _configure_jax_cache(cache_dir: str | None) -> None:
    """
    Set aggressive JAX persistent-cache flags and point the cache at
    `cache_dir`. Without `jax_compilation_cache_dir`, the persistent
    cache is disabled and every process restart pays the full compile
    cost — even when the on-disk flags below are tuned.

    Parameters
    ----------
    cache_dir : str | None
        Directory used by JAX as the persistent XLA cache. When `None`
        the cache is left at its default (effectively disabled).
    """
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    if cache_dir is not None:
        jax.config.update("jax_compilation_cache_dir", cache_dir)


def _resolve_chunk_size(num_trainers: int, max_group_size: int) -> int:
    """
    Largest divisor of `num_trainers` not exceeding `max_group_size`.

    Picking a divisor guarantees every chunk has identical shape, so
    `_batch_grad_fn` compiles a single XLA kernel rather than one per
    chunk-size variant.

    Parameters
    ----------
    num_trainers : int
        Total number of trainers to chunk.
    max_group_size : int
        Upper bound on the chunk size (memory budget proxy).

    Returns
    -------
    chunk_size : int
        The largest divisor of `num_trainers` such that
        `chunk_size <= min(max_group_size, num_trainers)`.
    """
    cap = min(max_group_size, num_trainers)

    for d in range(cap, 0, -1):
        if num_trainers % d == 0:
            return d

    return 1


class AgentTrainer:
    """
    Factory for a single continuous-action agent training slot in the `TrainerPool`.

    Creates `PolicyAgent`, target `PolicyAgent`, `DiscoValueAgent`, and
    optimizer states. After construction, `TrainerPool` extracts the
    agents, params, and state — this class serves as an initializer and
    is recreated on trainer resets. Environment stepping is handled by
    the `MultiVecEnv` held by `TrainerPool`, so this class does not
    own any env state.

    Lifetime
    ---------
    Each trainer is assigned a step budget on construction, sampled from the
    DiscoRL distribution `{20M, 50M, 100M, 200M}` environment steps with
    weights inversely proportional to budget size. When the budget is
    exhausted, `TrainerPool` detects the overspend and `RuleTrainer`
    replaces this trainer with a freshly initialized one before the next
    collection step. This ensures the meta-learner continually observes
    agents learning from scratch rather than converged policies.

    Parameters
    ----------
    env_name : str
        The registered envrax env name (e.g. `"mjx/hopper_hop-v0"`)
    config : AgentTrainerSettings
        Configuration for individual agent training
    key : chex.PRNGKey
        Random number generator key
    obs_space : envrax.Box
        Per-env observation space (typically from
        `multi_vec.single_observation_spaces[i]`)
    action_space : envrax.Box
        Per-env action space (typically from
        `multi_vec.single_action_spaces[i]`)
    budget_rng : random.Random
        Python RNG used to sample the initial step budget
    max_action_dim : int
        Maximum continuous action dimensionality across all environments
    max_obs_dim : int (optional)
        Maximum observation dimensionality across all environments.
        When provided, observations are zero-padded to this size and
        encoders use `max_obs_dim` as input width for `jax.vmap`
        compatibility. Default is `None`
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    """

    def __init__(
        self,
        env_name: str,
        config: AgentTrainerSettings,
        *,
        key: chex.PRNGKey,
        obs_space: envrax.Box,
        action_space: envrax.Box,
        budget_rng: random.Random,
        max_action_dim: int,
        max_obs_dim: int | None = None,
        jit_compile: bool = False,
    ) -> None:
        self.config = config
        self.env_name = env_name
        self.obs_space = obs_space
        self.action_space = action_space

        self.jit_compile = jit_compile
        self.key = key

        # Init state
        self.policy_optim = self._create_optim()
        self.value_optim = self._create_optim()

        self.ema_utils = MovingAverage(self.config.ema)

        self.meta_optim: optax.GradientTransformation = None  # type: ignore
        self.meta_opt_state: optax.OptState = None  # type: ignore

        # Lifetime tracking
        self._env_steps: int = 0
        self._step_budget: int = sample_budget(budget_rng)

        # Episode tracking
        self.episode_tracker = EpisodeTracker(1)
        self._collection_step = 0

        self.max_action_dim = max_action_dim
        self.max_obs_dim = max_obs_dim

        self.action_dim = math.prod(action_space.shape)
        self.key, agent_key, target_key, value_key = jax.random.split(key, 4)

        self.policy_agent = PolicyAgent(
            obs_space,
            action_space,
            config=self.config.agent,
            key=agent_key,
            max_action_dim=self.max_action_dim,
            max_obs_dim=self.max_obs_dim,
        )

        self.target_agent = PolicyAgent(
            obs_space,
            action_space,
            config=self.config.agent,
            key=target_key,
            max_action_dim=self.max_action_dim,
            max_obs_dim=self.max_obs_dim,
        )

        self.value_agent = DiscoValueAgent(
            obs_space,
            self.config.agent.n_hidden,
            config=self.config.value,
            key=value_key,
            sparsity=self.config.agent.sparsity,
            max_obs_dim=self.max_obs_dim,
        )

        self.state = AgentTrainerState.create(
            self.policy_optim.init(self.policy_agent.get_params()),
            self.value_optim.init(self.value_agent.get_params()),
        )

        # Pre-compute action dim mask
        self.action_dim_mask = jnp.arange(self.max_action_dim) < self.action_dim

    @property
    def n_actions(self) -> int:
        """Get number of actions."""
        return self.action_dim

    def warm(self) -> None:
        """
        Performs an initial forward pass through the agent networks.

        Includes -
            1. Materializing parameters before optimizer init
            2. JIT compile caching
        """
        obs_dim = self.max_obs_dim or math.prod(self.obs_space.shape)
        dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, 1, self.max_action_dim), dtype=jnp.float32)

        _, h_policy = self.policy_agent(dummy_obs, dummy_action)
        _, h_target = self.target_agent(dummy_obs, dummy_action)
        _, h_value = self.value_agent(dummy_obs)

        self.state = self.state.update_hidden(
            AgentTrainerHiddenStates(
                policy_ocm=h_policy.ocm,
                policy_acm=h_policy.acm,
                target_ocm=h_target.ocm,
                target_acm=h_target.acm,
                value=h_value,
            )
        )

    def compute_value_outs(
        self,
        rollout: Rollout,
        adv_ema: EMAState,
        td_ema: EMAState,
        action_dim_mask: jax.Array | None = None,
    ) -> Tuple[ValueOutputs, EMAState, EMAState]:
        """
        Compute value outputs using Gaussian importance weights.

        Parameters
        ----------
        rollout : Rollout
            Trajectory to compute value targets on
        adv_ema : EMAState
            Current advantage EMA state
        td_ema : EMAState
            Current TD-error EMA state
        action_dim_mask : jax.Array (optional)
            Boolean mask `(max_action_dim,)`. Default is `None`

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
            action_dim_mask=action_dim_mask,
        )

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


class RuleTrainer:
    """
    Discovers a Reinforcement Learning (RL) update rule by meta-training across
    a population of agents spanning diverse environments.

    `RuleTrainer` handles the outer loop of target rule learning (DiscoRL): managing
    a population of `AgentTrainer`s, computing meta-gradients from their learning
    progress, and updating the meta-network to improve collective agent performance.

    All trainers produce identically-shaped outputs (action-invariant architecture),
    enabling `jax.vmap` to compute meta-gradients for an entire chunk in one batched
    accelerator call. A `TrainerPool` holds all agent parameters, optimizer states,
    and rollout buffers as permanently stacked arrays — eliminating per-step
    stacking overhead and VRAM allocation cycles. Environment stepping is
    parallelized via a worker pool.

    Parameters
    ----------
    envs : envrax.EnvSuite | envrax.EnvSet
        Environment collection to train on. Pass a single suite
        (e.g. `DmControlSuite()`, or `DmControlSuite()[:3]` for a curated
        subset) or an `EnvSet` combining multiple suites.
    config : RuleTrainerSettings
        Configuration for meta-training
    agents_per_env : int (optional)
        Number of independent agent trainers to instantiate per environment.
        For example, if we have 25 environments and 2 agents per environment we have
        50 agent trainers. Formula: `n_agents = agents_per_env * n_envs`.

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
        Set to `None` to disable. Default is `".jax_cache"`

        On repeated runs with the same model shapes, compilation is skipped and
        loaded from disk instead. Only active when `jit_compile=True`.
    verbose : bool (optional)
        Dashboard display settings. Default is `True`.
            - If `True` - renders the full Rich dashboard
            - If `False` - uses a lightweight `tqdm` progress bar instead
    """

    def __init__(
        self,
        envs: envrax.EnvSuite | envrax.EnvSet,
        config: RuleTrainerSettings,
        *,
        agents_per_env: int = 2,
        max_group_size: int = 8,
        seed: int = 42,
        jit_compile: bool = True,
        cache_dir: str | None = ".jax_cache",
        verbose: bool = True,
    ) -> None:
        _configure_jax_cache(cache_dir)
        _cache_status = cache_status(cache_dir, jit_compile)

        if isinstance(envs, envrax.EnvSuite):
            envs = envrax.EnvSet(envs)

        envs.verify_packages()

        # Canonical name list + per-category counts
        self._unique_env_names = envs.all_names()
        self._env_categories = envs.env_categories()
        self._env_specs = sorted(self._unique_env_names * agents_per_env)

        self.env_names = list(self._env_specs)
        self.num_envs = len(self._unique_env_names)
        self.num_trainers = len(self._env_specs)

        self.n_steps = config.n_meta_steps(self.num_trainers)
        self._budget_rng = random.Random(seed)

        # Store params
        self.config = config
        self.agents_per_env = agents_per_env
        self.max_group_size = max_group_size

        self.jit_compile = jit_compile
        self.cache_dir = cache_dir

        self._seed = seed

        self._chunk_size = _resolve_chunk_size(self.num_trainers, max_group_size)

        # Init lineage
        self._parent_run: str | None = None
        self._parent_checkpoint_step: int | None = None

        # Init logger - one writer per unique env
        self.logger = MetricsLogger(self.config.run.log_dir)
        self.logger.add_writer("meta")
        self.logger.add_writer_group("envs", self._unique_env_names)

        # Init runtime logger - capture warnings and stderr to runtime.log
        self.runtime_logger = RuntimeLogger(self.config.run.dirpath)

        # Configure RNG keys
        key = jax.random.key(seed)
        self.key, meta_key, *self.trainer_keys = jax.random.split(
            key, self.num_trainers + 2
        )

        # Build multi-env — one single `JaxEnv` per trainer slot
        env_by_name: Dict[str, envrax.JaxEnv] = {
            name: envrax.make(
                name,
                jit_compile=self.jit_compile,
                pre_warm=False,
                cache_dir=self.cache_dir,
            )
            for name in self._unique_env_names
        }
        envs_by_trainer: Dict[str, envrax.JaxEnv] = {
            f"trainer_{i}": env_by_name[name] for i, name in enumerate(self._env_specs)
        }
        self.multi_env = envrax.make_multi(envs_by_trainer)

        # Ordered list of dict keys matching trainer order for slot lookups
        self._trainer_env_keys: List[str] = self.multi_env.env_keys

        # Padded shapes for `jax.vmap` over heterogeneous envs
        self.max_action_dim, _obs_dim = self.multi_env.pad_dims()
        self.max_obs_dim = _obs_dim or None

        self._batch_grad_fn: Callable[..., MetaGradOutput] = None  # type: ignore
        self._batched_collect_fn: Callable = None  # type: ignore

        self._meta_optim_update: Callable = None  # type: ignore

        # Pre-computed constants
        self._action_masks: Dict[int, jax.Array] = {}
        self._hidden_shapes: HiddenShapeCache = None  # type: ignore

        # Training pools
        self.pool: TrainerPool = None  # type: ignore

        # Init meta-agent and optimizer
        self.meta_agent = DiscoAgent(
            config=config.disco_agent,
            key=meta_key,
            max_action_dim=self.max_action_dim,
        )
        self.meta_optim = optax.adam(config.meta_lr)

        # Init disco diagnostics writers
        self.logger.add_writer_group("disco", self.meta_agent._disco_net.layer_names)

        # Init state
        self.state = RuleTrainerState.create(
            self.meta_optim.init(self.meta_agent.get_params()),
            self.num_trainers,
            self.config.batch_size,
            *self.meta_agent.hidden_sizes,
        )
        self.trainers: List[AgentTrainer] = []

        # Checkpointing
        self.cp_manager = CheckpointManager(self.config.run)

        # init console dashboard
        _n_chunks = self.num_trainers // self._chunk_size

        if verbose:
            self.console = DiscoConsoleDashboard(
                self.config.console_config(
                    envs=self._env_categories,
                    num_trainers=self.num_trainers,
                    n_chunks=_n_chunks,
                    params=self._dummy_params(self.trainer_keys[0]),
                    complete_path=str(self.config.run.dirpath),
                    jit_compile=jit_compile,
                    cache_status=_cache_status,
                )
            )
        else:
            self.console = SimpleDashboard(self.n_steps)

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
        first_key = self._trainer_env_keys[0]
        obs_space: envrax.Box = self.multi_env.observation_spaces[first_key]  # type: ignore
        action_space: envrax.Box = self.multi_env.action_spaces[first_key]  # type: ignore

        policy = PolicyAgent(
            obs_space,
            action_space,
            config=config.agent,
            key=rng_key,
            max_action_dim=self.max_action_dim,
            max_obs_dim=self.max_obs_dim,
        )

        value = DiscoValueAgent(
            obs_space,
            config.agent.n_hidden,
            config=config.value,
            key=rng_key,
            sparsity=config.agent.sparsity,
        )

        return DiscoParamsSettings(
            policy=policy.param_count,
            value=value.param_count,
            disco=self.meta_agent.param_count,
        )

    def _create_trainer(
        self,
        env_name: str,
        config: AgentTrainerSettings,
        *,
        key: chex.PRNGKey,
        obs_space: envrax.Box,
        action_space: envrax.Box,
    ) -> AgentTrainer:
        """
        Creates a single agent trainer.

        Parameters
        ----------
        env_name : str
            The registered envrax environment name (e.g. `"mjx/hopper_hop-v0"`)
        config : AgentTrainerSettings
            Configuration for individual agent training
        key : chex.PRNGKey
            Random number generator key
        obs_space : envrax.Box
            Per-env observation space (from `multi_vec.single_observation_spaces[i]`)
        action_space : envrax.Box
            Per-env action space (from `multi_vec.single_action_spaces[i]`)

        Returns
        -------
        trainer : AgentTrainer
            Initialized agent trainer
        """
        return AgentTrainer(
            env_name,
            config,
            key=key,
            obs_space=obs_space,
            action_space=action_space,
            budget_rng=self._budget_rng,
            max_action_dim=self.max_action_dim,
            max_obs_dim=self.max_obs_dim,
            jit_compile=self.jit_compile,
        )

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
        setup_total = 2 * self.num_trainers + 7
        self.console.start_setup(setup_total)

        # Build trainers
        for i, env_name in enumerate(self._env_specs):
            key = self._trainer_env_keys[i]
            trainer = self._create_trainer(
                env_name,
                self.config.agent_trainer_config(),
                key=trainer_keys[i],
                obs_space=self.multi_env.observation_spaces[key],  # type: ignore
                action_space=self.multi_env.action_spaces[key],  # type: ignore
            )
            self.trainers.append(trainer)
            self.console.update_setup()

        # Init meta-optimizers and warm trainers
        for trainer in self.trainers:
            trainer._init_meta_optim(self.meta_agent.get_params())
            trainer.warm()
            self.console.update_setup()

        # Pre-compute shared constants for the inner loop
        self._action_masks = {t.action_dim: t.action_dim_mask for t in self.trainers}

        # Cache hidden state shapes for batched collection (None → zeros)
        h0 = self.trainers[0].state.hidden
        self._hidden_shapes = HiddenShapeCache(
            policy_ocm=jnp.shape(h0.policy_ocm),  # type: ignore
            policy_acm=jnp.shape(h0.policy_acm),  # type: ignore
            target_ocm=jnp.shape(h0.target_ocm),  # type: ignore
            target_acm=jnp.shape(h0.target_acm),  # type: ignore
            value=jnp.shape(h0.value),  # type: ignore
        )

        # Build functions
        self._batch_grad_fn = self._build_batch_grad_fn()
        self._batched_collect_fn = self._build_batched_collect_fn()
        self.console.update_setup()

        # Compile each unique inner JaxEnv up front; sharing by name
        # means P=N_unique compiles, not P=N_trainers.
        self.multi_env.compile(progress=False)

        # Build trainer pool — stacks all params/states as permanent GPU arrays
        self.key, pool_rng = jax.random.split(self.key)
        self.pool = TrainerPool(
            self.trainers,
            max_action_dim=self.max_action_dim,
            action_masks=self._action_masks,
            hidden_shapes=self._hidden_shapes,
            n_updates=self.config.n_updates,
            seq_len=self.config.seq_len,
            encoding_dim=self.trainers[0].policy_agent.encoding_dim,
            prediction_dim=self.config.agent.prediction_size,
            max_obs_dim=self.max_obs_dim or 0,
            multi_env=self.multi_env,
            replay_capacity=self.config.replay_capacity,
            replay_ratio=self.config.replay_ratio,
            init_rng=pool_rng,
        )
        self.console.update_setup()

        # Pre-compile all JAX kernels so train() starts warm
        self._meta_optim_update = WarmCompile(self).run()

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
            ),
        )

    def _build_batched_collect_fn(self) -> Callable:
        """
        Build a JIT + vmap compiled forward pass for batched collection.

        Processes all trainers in one GPU call by vmapping a pure forward
        function over stacked params, observations, and hidden states.
        Uses the first trainer as a template for graph definitions.

        Returns
        -------
        fn : Callable
            JIT + vmap compiled collection forward function
        """
        t = self.trainers[0]  # template

        def _pure_forward(
            obs,
            prev_actions,
            p_params,
            t_params,
            v_params,
            p_ocm_h,
            p_acm_h,
            t_ocm_h,
            t_acm_h,
            v_h,
            a_dim_mask,
        ):
            # Reconstruct all modules from explicit params
            encoder, p_ocm, p_acm, p_dec = t.policy_agent.merge_params(p_params)
            _, t_ocm, t_acm, t_dec = t.target_agent.merge_params(t_params)
            v_net = t.value_agent.merge_params(v_params).net

            # Shared encoding
            encoding = encoder(obs)

            # Add time dimension to match encoding shape from encoder
            if prev_actions.ndim == encoding.ndim - 1:
                prev_actions = jnp.expand_dims(prev_actions, axis=1)

            # Policy forward
            ocm_preds, new_p_ocm_h = p_ocm(encoding, h_state=p_ocm_h)
            acm_preds, new_p_acm_h = p_acm(
                ocm_preds.embedding,
                prev_actions,
                h_state=p_acm_h,
            )
            mu, log_std, aux_pi = p_dec(
                ocm_preds.pi,
                acm_preds.aux_pi,
                action_dim_mask=a_dim_mask,
            )
            preds = PolicyAgentOutput.create(
                encoding,
                mu,
                log_std,
                ocm_preds.y,
                acm_preds.z,
                aux_pi,
                acm_preds.q,
            )

            # Target forward
            t_ocm_preds, new_t_ocm_h = t_ocm(encoding, h_state=t_ocm_h)
            t_acm_preds, new_t_acm_h = t_acm(
                t_ocm_preds.embedding,
                prev_actions,
                h_state=t_acm_h,
            )
            t_mu, t_log_std, t_aux_pi = t_dec(
                t_ocm_preds.pi,
                t_acm_preds.aux_pi,
                action_dim_mask=a_dim_mask,
            )
            target_preds = PolicyAgentOutput.create(
                encoding,
                t_mu,
                t_log_std,
                t_ocm_preds.y,
                t_acm_preds.z,
                t_aux_pi,
                t_acm_preds.q,
            )

            # Value forward
            v, new_v_h = v_net(encoding, h_state=v_h)
            values = squeeze_time(v)

            return (
                preds,
                target_preds,
                values,
                new_p_ocm_h,
                new_p_acm_h,
                new_t_ocm_h,
                new_t_acm_h,
                new_v_h,
            )

        return jax.jit(jax.vmap(_pure_forward))

    def _compute_inner_policy_loss(
        self,
        trainer: AgentTrainer,
        p_params: nnx.State,
        encoding: jax.Array,
        actions: jax.Array,
        targets: DiscoAgentOutput,
        discounts: jax.Array,
        mask: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Compute inner-loop policy loss for a single continuous update step.

        Runs a functional forward pass through the continuous policy agent
        with explicit parameters and computes the Gaussian policy loss against
        disco targets.

        Parameters
        ----------
        trainer : AgentTrainer
            Read-only reference for functional forward
        p_params : flax.nnx.State
            Policy network parameters (differentiated by caller)
        encoding : jax.Array
            Encoder embeddings `(B, T, F)`
        actions : jax.Array
            Continuous actions taken `(B, T, A)`
        targets : DiscoAgentOutput
            Disco targets for this update step
        discounts : jax.Array
            Episode discount factors `(B, T)`
        mask : jax.Array
            Boolean mask `(max_action_dim,)` for valid action dimensions

        Returns
        -------
        total_loss : jax.Array
            Total weighted policy loss
        pi_loss : jax.Array
            Policy KL divergence component
        """
        new_preds = trainer.policy_agent.functional_forward(
            encoding,
            actions,
            p_params,
            action_dim_mask=mask,
        )
        return compute_policy_loss(
            targets,
            new_preds.mu,
            new_preds.log_std,
            new_preds.y,
            new_preds.z,
            new_preds.aux_pi,
            discounts,
            self.config.loss_cost,
            mask,
        )

    def _compute_outer_losses(
        self,
        valid_rollout: Rollout,
        adv: jax.Array,
        scan_aux: Tuple[DiscoAgentOutput, jax.Array, jax.Array],
        mask: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute validation policy gradient, entropy, and regularization losses
        for continuous action spaces.

        Parameters
        ----------
        valid_rollout : Rollout
            Validation rollout `(B, T, ...)`
        adv : jax.Array
            Stop-gradient normalized advantages from validation
        scan_aux : Tuple[DiscoAgentOutput, jax.Array, jax.Array]
            Accumulated `(targets, target_mu, target_log_std)` from
            inner-loop scan steps
        mask : jax.Array
            Boolean mask `(max_action_dim,)` for valid action dimensions

        Returns
        -------
        pg_loss : jax.Array
            Gaussian policy gradient loss
        entropy_loss : jax.Array
            Gaussian entropy regularization loss
        reg_loss : jax.Array
            Target regularization loss
        """
        all_targets, all_target_mu, all_target_log_std = scan_aux

        pg_loss = compute_gaussian_policy_gradient_loss(
            valid_rollout.preds.mu,
            valid_rollout.preds.log_std,
            valid_rollout.actions,
            adv,
            action_dim_mask=mask,
        ).mean()
        entropy_loss = compute_gaussian_entropy_loss(
            valid_rollout.preds.log_std,
            self.config.entropy_coef,
            action_dim_mask=mask,
        )
        reg_loss = compute_meta_reg_loss(
            jax.tree.map(lambda x: x[-1], all_targets),
            jax.tree.map(lambda x: x[-1], all_target_mu),
            jax.tree.map(lambda x: x[-1], all_target_log_std),
            self.config.reg_scale,
            self.config.kl_reg,
            action_dim_mask=mask,
        )
        return pg_loss, entropy_loss, reg_loss

    def _compute_meta_gradient(
        self,
        trainer: AgentTrainer,
        meta_params: chex.ArrayTree,
        p_params: chex.ArrayTree,
        v_params: chex.ArrayTree,
        disco_h: jax.Array,
        meta_h: jax.Array,
        p_opt_state: chex.ArrayTree,
        v_opt_state: chex.ArrayTree,
        adv_ema: EMAState,
        td_ema: EMAState,
        train_rollouts: Rollout,
        valid_rollout: Rollout,
        action_mask: jax.Array,
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
        disco_h : jax.Array
            Disco network hidden state `(B, H)`
        meta_h : jax.Array
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
        action_mask : jax.Array
            Boolean mask `(max_actions,)` for valid actions

        Returns
        -------
        out : MetaGradOutput
            Gradient and updated state outputs
        """

        def meta_loss_fn(
            meta_params,
        ) -> Tuple[jax.Array, Tuple[MetaLossAux, EMAState, EMAState]]:
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

                def inner_policy_loss(p, enc, act, tgt, disc):
                    return self._compute_inner_policy_loss(
                        trainer,
                        p,
                        enc,
                        act,
                        tgt,
                        disc,
                        action_mask,
                    )

                (_, _), p_grads = fwdrev_value_and_grad(
                    inner_policy_loss,
                    has_aux=True,
                )(carry.p_params, encoding, actions, targets, discounts)

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
                    net_out = value_outs.value[:-1]
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
                return new_carry, (
                    targets,
                    rollout.target_preds.mu,
                    rollout.target_preds.log_std,
                )

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

            final_out, scan_aux = jax.lax.scan(
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

            pg_loss, entropy_loss, reg_loss = self._compute_outer_losses(
                valid_rollout,
                adv,
                scan_aux,
                action_mask,
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

    def _get_rollouts(self) -> None:
        """
        Collect all rollouts into the pool's buffers using batched
        accelerator forward passes.

        Handles trainer resets, runs the training collection phase,
        updates step budgets and episode logging, performs soft target
        updates, then runs the validation collection phase.

        Training rollouts land in `pool.train_buffer` (a `MixedBuffer`
        ring); the validation rollout overwrites `pool.valid_rollout`.
        """
        # Handle resets before collection
        for idx in range(self.num_trainers):
            if self.pool.needs_reset(idx):
                self.reset_trainer(idx)
                self.pool.reset_trainer(idx, self.trainers[idx])

        # Reset episode trackers — buffer write cursor lives in the
        # scan carry now, so no per-call buffer reset is needed.
        for tracker in self.pool.episode_trackers:
            tracker.reset()

        # Training: collection phase
        self.pool.collect(
            self._batched_collect_fn,
            n_rollouts=self.config.n_updates,
            seq_len=self.config.seq_len,
            training=True,
        )

        # Finalize: step budgets, episode logging
        self.pool.finalize_training(self.config.n_updates, self.config.seq_len)

        for i in range(self.num_trainers):
            tracker = self.pool.episode_trackers[i]

            if metrics := tracker.metrics():
                self.logger.log(
                    f"envs/{self.pool.env_names[i]}",
                    self.state.meta_step,
                    metrics,
                )

        # Soft update targets
        self.pool.soft_update_targets(self.config.tau)

        # Validation: collection phase
        self.pool.collect(
            self._batched_collect_fn,
            n_rollouts=1,
            seq_len=self.config.seq_len * 2,
            training=False,
        )

    def _process_chunk(
        self,
        start: int,
        end: int,
        meta_params: chex.ArrayTree,
        accumulated_grad: chex.ArrayTree,
        stats: MetaStepStats,
    ) -> chex.ArrayTree:
        """
        Compute meta-gradients for a chunk of trainers.

        Slices directly into the pool's stacked arrays — no intermediate
        stacking. Writes results back via in-place pool updates.

        Parameters
        ----------
        start : int
            First trainer index (inclusive)
        end : int
            Last trainer index (exclusive)
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
        # Slice disco/meta hidden states for this chunk
        stacked_disco_h, stacked_meta_h = [], []

        for idx in range(start, end):
            d_h, m_h = self.state.hidden.get(idx)
            stacked_disco_h.append(d_h)
            stacked_meta_h.append(m_h)

        stacked_disco_h = jnp.stack(stacked_disco_h)
        stacked_meta_h = jnp.stack(stacked_meta_h)

        # Get all gradient inputs from pool — split rng for replay sampling
        self.key, sample_rng = jax.random.split(self.key)
        grad_inputs = self.pool.get_grad_inputs(
            start,
            end,
            stacked_disco_h,
            stacked_meta_h,
            rng=sample_rng,
            n_updates=self.config.n_updates,
            batch_size=self.config.batch_size,
        )

        # Single accelerator call for entire chunk
        chunk_out: MetaGradOutput = self._batch_grad_fn(
            meta_params,
            *grad_inputs,
        )
        del grad_inputs, stacked_disco_h, stacked_meta_h

        # Populate gradient params/states back into pool
        self.pool.update_from_grad(start, chunk_out)

        # Only transfer scalar logging data to CPU
        log_data: ChunkLogData = jax.device_get(chunk_out.log_data())

        # Vmapped meta-optim update over the whole chunk — one jit call
        chunk_opt = jax.tree.map(lambda x: x[start:end], self.pool.meta_opt_state)
        normed_grads_stack, new_chunk_opt = jax.vmap(
            self.pool.meta_optim_update, in_axes=(0, 0, None)
        )(chunk_out.meta_grad, chunk_opt, meta_params)

        # Scatter updated chunk back into the stacked pool state
        self.pool.meta_opt_state = jax.tree.map(
            lambda dst, src: dst.at[start:end].set(src),
            self.pool.meta_opt_state,
            new_chunk_opt,
        )

        # Sum per-trainer normed gradients into the running meta-grad accumulator
        accumulated_grad = jax.tree.map(
            lambda acc, g: acc + g.sum(axis=0),
            accumulated_grad,
            normed_grads_stack,
        )

        # Per-trainer host-side bookkeeping (hidden state + metric logging)
        chunk_size = end - start
        for i in range(chunk_size):
            idx = start + i

            self.state = self.state.update_hidden(
                idx,
                chunk_out.disco_h[i].copy(),
                chunk_out.meta_h[i].copy(),
            )

            self._log_meta_metrics(
                idx,
                log_data.pg_loss[i],
                log_data.entropy_loss[i],
                log_data.reg_loss[i],
                log_data.meta_loss[i],
                log_data.advantages[i],
                log_data.normalized_advantages[i],
            )

            losses_i = LossStatistics(
                meta=log_data.meta_loss[i],
                policy_gradient=log_data.pg_loss[i],
                entropy=log_data.entropy_loss[i],
                regularization=log_data.reg_loss[i],
            )

            ep_tracker: EpisodeTracker = self.pool.episode_trackers[idx]
            stats.record(
                ep_tracker.windowed_mean_return,
                ep_tracker.windowed_mean_length,
                losses_i,
            )

        # Accelerator cleanup
        del chunk_out, log_data
        jax.effects_barrier()  # flush async ops

        return accumulated_grad

    def train(self) -> None:
        """
        Performs meta-training loop to discover an RL update rule.

        Each meta-step collects rollouts into the pool's buffers
        via batched vmapped forward passes, then computes meta-gradients
        in chunks on the accelerator by slicing directly into the pool's
        stacked arrays.

        Includes
        --------
        1. For each meta-step:

            a. Collect rollouts from all trainers (batched forward)
            b. Process trainers in chunks via vmapped gradient computation
            c. Accumulate meta-gradients through the learning process
            d. Average gradients and update meta-network

        2. Log metrics and checkpoints periodically
        """
        # Setup - build trainers, warm networks, compile gradient functions
        self._initial_setup(self.trainer_keys)
        self.console.finish_setup()

        self.console.start_training()

        try:
            for _ in range(self.state.meta_step, self.n_steps):
                stats = MetaStepStats(self.num_trainers)
                meta_params = self.meta_agent.get_params()

                accumulated_grad = jax.tree.map(jnp.zeros_like, meta_params)

                self._get_rollouts()
                jax.effects_barrier()  # flush async ops

                # Process all chunks — uniform size via `_chunk_size`
                for start in range(0, self.num_trainers, self._chunk_size):
                    end = start + self._chunk_size
                    chunk_envs = self.pool.env_names[start:end]

                    self.console.update_progress(
                        "Inner Updates",
                        chunk_size=end - start,
                        env_names=chunk_envs,
                    )

                    # Compute gradients for this chunk
                    accumulated_grad = self._process_chunk(
                        start,
                        end,
                        meta_params,
                        accumulated_grad,
                        stats,
                    )

                # Apply average gradients through single shared optimizer
                avg_grad = jax.tree.map(
                    lambda g: g / self.num_trainers,
                    accumulated_grad,
                )
                del accumulated_grad  # Free before meta update allocates
                self._apply_meta_update(avg_grad)

                # Log metrics
                grad_norm = float(optax.global_norm(avg_grad))

                metrics = {"meta/grad_norm": grad_norm}
                self.logger.log("meta", self.state.meta_step, metrics)
                self._compute_diagnostics()

                self.console.update_stats(**stats.rewards_as_dict())
                self.console.update_losses(
                    **stats.losses_as_dict(),
                    gradient_norm=grad_norm,
                )

                # Checkpoint periodically
                self.save_checkpoint()
                self.console.update_progress("Meta Steps")

                # Cleanup
                del avg_grad, stats
                jax.effects_barrier()

        except (KeyboardInterrupt, SystemExit):
            exit()
        finally:
            # Save last step if not already saved
            if self.cp_manager.latest_step != self.state.meta_step:
                self.save_checkpoint(force=True)

            self.close()

        self.console.finish_training()

    @classmethod
    def restore(
        cls,
        run_dir: str,
        *,
        checkpoint_step: int | None = None,
        total_env_steps: int | None = None,
        run: RunSettings | None = None,
        max_group_size: int = 8,
        jit_compile: bool = True,
        cache_dir: str | None = ".jax_cache",
        verbose: bool = True,
    ) -> Self:
        """
        Restore meta-agent parameters from a previous run and create a
        fresh training run.

        Creates a new run directory with fresh checkpoints, logs, and
        training state (`meta_step=0`). Only the meta-agent parameters
        are carried over from the parent run. The parent run is left
        untouched.

        Call `train()` after to start training with the restored parameters.

        Parameters
        ----------
        run_dir : str
            Path to the parent run directory containing `metadata.json`
            and `checkpoints/`
        checkpoint_step : int (optional)
            Specific checkpoint step to restore from the parent run.
            When `None`, restores the latest checkpoint. Default is `None`
        total_env_steps : int (optional)
            Total environment step budget for the new run. When `None`,
            uses the same `total_env_steps` from the parent config.
            Default is `None`
        run : RunSettings (optional)
            Run directory settings for the new run. When `None`, creates
            a fresh `RunSettings()` with default values. Default is `None`
        max_group_size : int (optional)
            Maximum number of trainers to `vmap` simultaneously. Higher values
            improve GPU utilization but increase VRAM usage. Reduce if
            out of memory (OOM). Default is `8`
        jit_compile : bool (optional)
            Flag to enable/disable JIT compilation. Default is `True`
        cache_dir : str | None (optional)
            Directory path for JAX's persistent XLA compilation cache.
            Set to `None` to disable. Default is `".jax_cache"`
        verbose : bool (optional)
            Dashboard display settings. Default is `True`.
                - If `True` - renders the full Rich dashboard
                - If `False` - uses a lightweight `tqdm` progress bar instead

        Returns
        -------
        trainer : RuleTrainer
            A fresh `RuleTrainer` with restored meta-agent parameters

        Raises
        ------
        file_error : FileNotFoundError
            If `metadata.json` is not found in `run_dir`
        checkpoint_error : ValueError
            If no checkpoint exists in `run_dir`
        """
        print("Restoring model... ", end="")
        parent_settings = RunSettings.from_path(run_dir)

        # Load metadata from the parent run
        with CheckpointManager(parent_settings) as parent_manager:
            meta_run = parent_manager.load_metadata(RuleTrainerMetadata)
            restored_step = checkpoint_step or parent_manager.latest_step

            # Reconstruct env name list and config from saved metadata
            config: RuleTrainerSettings = load_config(
                RuleTrainerSettings,
                meta_run.config,
            )

            # Fresh run directory with optional overrides
            config = config.__replace__(
                run=run if run is not None else RunSettings(),
                total_env_steps=(
                    total_env_steps
                    if total_env_steps is not None
                    else config.total_env_steps
                ),
            )

            # Construct fresh trainer (new run dir, meta_step=0)
            trainer = cls(
                envrax.EnvSet.from_names(meta_run.env_names),
                config=config,
                agents_per_env=meta_run.agents_per_env,
                max_group_size=max_group_size,
                seed=meta_run.seed,
                jit_compile=jit_compile,
                cache_dir=cache_dir,
                verbose=verbose,
            )

            # Restore meta-agent params from parent checkpoint (fresh state)
            trainer.load_checkpoint(
                step=checkpoint_step,
                params_only=True,
                source=parent_manager,
            )

        # Track lineage
        trainer._parent_run = str(parent_settings.dirpath)
        trainer._parent_checkpoint_step = int(restored_step or 0)

        print("Complete.")
        return trainer

    def _log_meta_metrics(
        self,
        trainer_idx: int,
        pg_loss: jax.Array,
        entropy_loss: jax.Array,
        reg_loss: jax.Array,
        total_loss: jax.Array,
        advantages: jax.Array,
        normalized_advantages: jax.Array,
    ) -> None:
        """
        Log meta-training metrics for a single trainer.

        Parameters
        ----------
        trainer_idx : int
            Index of the current agent trainer
        pg_loss : jax.Array
            Policy gradient loss
        entropy_loss : jax.Array
            Entropy regularization loss
        reg_loss : jax.Array
            L2 and KL regularization loss on meta-network targets
        total_loss : jax.Array
            Sum of all loss components
        advantages : jax.Array
            Raw advantages from validation rollout
        normalized_advantages : jax.Array
            EMA-normalised advantages from the validation rollout
        """
        metric_tree = {
            "meta/pg_loss": pg_loss,
            "meta/entropy_loss": entropy_loss,
            "meta/reg_loss": reg_loss,
            "meta/total_loss": total_loss,
            "meta/advantages": jnp.mean(advantages),
            "meta/normalized_advantages": jnp.mean(normalized_advantages),
        }
        metrics: Dict[str, float] = jax.device_get(metric_tree)
        self.logger.log(
            f"envs/{self.env_names[trainer_idx]}",
            self.state.meta_step,
            metrics,
        )

    def _compute_diagnostics(self) -> None:
        """
        Compute DiscoNetwork cell diagnostics averaged across all trainers.

        Stacks per-trainer disco hidden states into one `(P, B, H)`
        tensor and vmaps `diagnostics` across the population — one JAX
        dispatch instead of `num_trainers` sequential ones. Averages
        scalar outputs over the population axis and logs to per-layer
        TensorBoard writers under `disco/`.
        """
        disco_in_features = self.meta_agent._disco_net.in_features

        stacked_h = jnp.stack(
            [self.state.hidden.get(i)[0] for i in range(self.num_trainers)]
        )  # (P, B, H_disco)
        P = stacked_h.shape[0]
        batch_size = stacked_h.shape[1]
        x = jnp.zeros((P, batch_size, disco_in_features))

        diag_stacked = jax.vmap(self.meta_agent._disco_net.diagnostics)(
            x, stacked_h
        )
        averaged_tree = jax.tree.map(lambda v: jnp.mean(v), diag_stacked)
        averaged: Dict[str, float] = jax.device_get(averaged_tree)

        self.logger.log_group("disco", self.state.meta_step, averaged)

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
            - `metadata.json` - config and environment set

        Parameters
        ----------
        force : bool (optional)
            Force save even if within save interval. Default is `False`

        Returns
        -------
        saved : bool
            Whether checkpoint was actually saved
        """
        checkpoint = {
            "meta_params": self.meta_agent.get_params(),
            "state": self.state,
        }

        metadata = RuleTrainerMetadata(
            config=dump_config(self.config),
            agents_per_env=self.agents_per_env,
            max_group_size=self.max_group_size,
            seed=self._seed,
            env_names=self._unique_env_names,
            env_categories=self._env_categories,
            disco_key=get_rng_key_data(self.meta_agent.key),
            parent_run=self._parent_run,
            parent_checkpoint_step=self._parent_checkpoint_step,
        )

        return self.cp_manager.save(
            self.state.meta_step,
            checkpoint,
            metadata=metadata,
            force=force,
        )

    def load_checkpoint(
        self,
        step: int | None = None,
        *,
        params_only: bool = False,
        source: CheckpointManager | None = None,
    ) -> None:
        """
        Restore training state from a checkpoint.

        Parameters
        ----------
        step : int (optional)
            Specific step to restore, or `None` for latest.
            Default is `None`
        params_only : bool (optional)
            When `True`, only restores meta-agent parameters and keeps
            the current training state (optimizer, hidden states,
            `meta_step`). Used when extending training in a fresh run.
            Default is `False`
        source : CheckpointManager (optional)
            Checkpoint manager to restore from. When `None`, uses
            `self.cp_manager`. Default is `None`

        Raises
        ------
        checkpoint_error : ValueError
            If no checkpoint exists in the checkpoint directory
        """
        manager = source if source is not None else self.cp_manager

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

        restored = manager.restore(step, abstract_checkpoint)

        # Restore meta-agent params (and optionally full state)
        self.meta_agent.update_params(restored["meta_params"])

        if not params_only:
            self.state = restored["state"]

        # Reset all agent trainers (skip if not yet built — train() handles that)
        for idx in range(len(self.trainers)):
            self.reset_trainer(idx)

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
        env_name = self._env_specs[trainer_idx]

        # Fresh RNG keys
        self.key, new_key = jax.random.split(self.key, 2)

        # Init new trainer
        key = self._trainer_env_keys[trainer_idx]
        new_trainer = self._create_trainer(
            env_name,
            self.config.agent_trainer_config(),
            key=new_key,
            obs_space=self.multi_env.observation_spaces[key],  # type: ignore[arg-type]
            action_space=self.multi_env.action_spaces[key],  # type: ignore[arg-type]
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
        self.logger.close()
        self.runtime_logger.close()
