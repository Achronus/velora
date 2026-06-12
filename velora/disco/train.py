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

import envrax
import jax
import optax

from velora.cli.disco.dashboard import DiscoConsoleDashboard
from velora.cli.disco.simple import SimpleDashboard
from velora.disco.config.metadata import PoolMetadata, RuleTrainerMetadata, SlotMetadata
from velora.disco.config.settings import RuleTrainerSettings
from velora.disco.factory import create_pool_state, create_specs, create_trainer_slot
from velora.utils.config import dump_config
from velora.utils.format import cache_status


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
        jax.config.update("jax_compilation_cache_dir", cache_dir)


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
        budget, giving the meta-learner diverse gradient signals. This enables agents
        with different learning stages and random initializations. Default is `2`
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
        seed: int = 42,
        jit_compile: bool = True,
        cache_dir: str | None = ".jax_cache",
        verbose: bool = True,
    ) -> None:
        # Initial error handling/verification steps
        _configure_jax_cache(cache_dir)
        _cache_status = cache_status(cache_dir, jit_compile)

        if isinstance(envs, envrax.EnvSuite):
            envs = envrax.EnvSet(envs)

        envs.verify_packages()

        # Store static info
        self.config = config

        self.jit_compile = jit_compile
        self.cache_dir = cache_dir

        # Set metadata
        self.metadata = self._create_rt_metadata(config, envs, agents_per_env, seed)

        # Configure RNG keys
        key = jax.random.key(seed)
        self.key, spec_key, pool_key, *self.trainer_keys = jax.random.split(
            key,
            self.metadata.num_trainers + 3,
        )

        # Build multi-env
        self.multi_env = self._create_multi_env()
        max_action_dim, max_obs_dim = self.multi_env.pad_dims()

        # Init disco optimizer
        self.meta_optim = optax.adam(config.meta_lr)

        # Init state, specs and trainer pool
        specs, disco_params = create_specs(
            (max_obs_dim,),
            max_action_dim,
            max_obs_dim,
            config,
            key=spec_key,
        )

        agent_states = [
            create_trainer_slot(
                env.observation_space,
                env.action_space,
                config=config,
                key=self.trainer_keys[i],
                specs=specs,
                disco_params=disco_params,
                max_action_dim=max_action_dim,
                max_obs_dim=max_obs_dim,
            )
            for i, env in enumerate(self.multi_env.envs.values())
        ]

        self.multi_env.env_keys

        self._pool_meta = self._create_pool_metadata(max_obs_dim, max_action_dim)
        self.pool = create_pool_state(
            agent_states,
            self.multi_env,
            self._pool_meta,
            config,
            specs.encoder,
            key=pool_key,
        )

        # init console dashboard
        if verbose:
            self.console = DiscoConsoleDashboard(
                self.config.console_config(
                    self.metadata,
                    params=self._dummy_params(self.trainer_keys[0]),
                    jit_compile=jit_compile,
                    cache_status=_cache_status,
                )
            )
        else:
            self.console = SimpleDashboard(self.metadata.n_meta_steps)

    def _create_rt_metadata(
        self,
        config: RuleTrainerSettings,
        envs: envrax.EnvSet,
        agents_per_env: int,
        seed: int,
    ) -> RuleTrainerMetadata:
        """
        Helper method. Create the rule trainer metadata and store it in an object.

        Parameters
        ----------
        config : RuleTrainerSettings
            A set of configuration settings
        envs : envrax.EnvSet
            Set of environments used during training
        agents_per_env : int
            Number of agents per environment
        seed : int
            Random number generator seed

        Returns
        -------
        metadata : RuleTrainerMetadata
            Rule trainer metadata
        """
        conf = dump_config(config)

        unique_env_names = envs.all_names()
        env_categories = envs.env_categories()

        num_envs = len(unique_env_names)
        num_trainers = num_envs * agents_per_env

        n_steps = config.n_meta_steps(num_trainers)

        return RuleTrainerMetadata(
            config=conf,
            num_envs=num_envs,
            num_trainers=num_trainers,
            agents_per_env=agents_per_env,
            n_meta_steps=n_steps,
            env_names=unique_env_names,
            env_categories=env_categories,
            seed=seed,
        )

    def _create_pool_metadata(
        self, max_obs_dim: int, max_action_dim: int
    ) -> PoolMetadata:
        """
        Helper method. Create the trainer pool metadata and store it in an object.

        Parameters
        ----------
        max_obs_dim : int
            Maximum observation size
        max_action_dim : int
            Maximum action size

        Returns
        -------
        pool_meta : PoolMetadata
            Trainer pool metadata
        """
        return PoolMetadata(
            slots=tuple(
                SlotMetadata(
                    env_key=k,
                    env_name=self.multi_env.envs[k].name,
                    obs_dim=self.multi_env.observation_sizes[k],
                    action_dim=self.multi_env.action_sizes[k],
                )
                for k in self.multi_env.env_keys
            ),
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )

    def _create_multi_env(self) -> envrax.MultiEnv:
        """
        Helper method. Create a multi-environment instance for managing
        multiple agent trainers.

        Returns
        -------
        multi_env : envrax.MultiEnv
            Multi-environment instance
        """
        _envs_list = [
            envrax.make(
                name,
                jit_compile=False,
                pre_warm=False,
                cache_dir=self.cache_dir,
            )
            for name in sorted(self.metadata.env_names * self.metadata.agents_per_env)
        ]
        return envrax.make_multi(
            {f"trainer_{i}": env for i, env in enumerate(_envs_list)}
        )
