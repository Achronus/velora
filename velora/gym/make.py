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

import gymnasium as gym
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from velora.gym.error import MissingPackageError


def make_mujoco_env(
    name: str,
    num_envs: int = 4,
    max_episode_steps: int = 1000,
    render_mode: str = "rgb_array",
    **kwargs,
) -> VectorEnv:
    """
    Creates a sync vectorized [MuJoCo](https://mujoco.org/) environment.

    Applies wrappers -
    - `gymnasium.wrappers.vector.RecordEpisodeStatistics`

    Parameters
    ----------
    name : str
        Name of the environment (e.g., `Ant-v5`)
    num_envs : int (optional)
        The number of vectorized environments to make. Default is `4`
    max_episode_steps : int (optional)
        Maximum number of episode steps. Default is `1000`
    render_mode : str (optional)
        The type of render mode for the environment.
        Default is `rgb_array`
    kwargs : Any (optional)
        Additional arguments passed to `gym.make_vec()`

    Returns
    -------
    envs : VectorEnv
        A set of wrapped vectorized environments
    """
    envs = gym.make_vec(
        name,
        num_envs=num_envs,
        vectorization_mode="sync",
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )
    return RecordEpisodeStatistics(envs)


def make_dmc_env(
    name: str,
    num_envs: int = 4,
    max_episode_steps: int = 1000,
    render_mode: str = "rgb_array",
    **kwargs,
) -> VectorEnv:
    """
    Creates a sync vectorized
    [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)
    environment.

    Loads each task via `dm_control.suite.load` directly to bypass
    `dm_control.locomotion` and its `labmaze` dependency (which has no
    Python 3.13 wheels). Wraps each task in
    `velora.gym.dm_control.DmControlSuiteEnv` for Gymnasium compatibility.

    Applies wrappers -
    - `velora.gym.dm_control.DmControlSuiteEnv`
    - `gymnasium.wrappers.TimeLimit`
    - `gymnasium.wrappers.vector.RecordEpisodeStatistics`

    Parameters
    ----------
    name : str
        Environment name in `dm_control/{domain}-{task}-v0` format
        (e.g., `dm_control/acrobot-swingup-v0`)
    num_envs : int (optional)
        The number of vectorized environments to make. Default is `4`
    max_episode_steps : int (optional)
        Maximum number of episode steps. Default is `1000`
    render_mode : str (optional)
        The type of render mode for the environment.
        Default is `rgb_array`
    kwargs : Any (optional)
        Additional arguments

    Returns
    -------
    envs : VectorEnv
        A set of wrapped vectorized environments
    """
    try:
        import dm_control.suite  # type: ignore
    except ImportError:
        raise MissingPackageError(
            "DMC environments require 'dm_control'. "
            "Install with: pip install dm_control"
        )

    from velora.gym.dm_control.compat import DmControlSuiteEnv

    env_id = name
    if env_id.startswith("dm_control/"):
        env_id = env_id[len("dm_control/") :]
    if env_id.endswith("-v0"):
        env_id = env_id[: -len("-v0")]

    parts = env_id.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid DMC environment name: {name}")

    domain, task = parts

    def _make_single_env():
        env = dm_control.suite.load(domain_name=domain, task_name=task)
        wrapped = DmControlSuiteEnv(env, render_mode=render_mode, **kwargs)
        return TimeLimit(wrapped, max_episode_steps=max_episode_steps)

    env_fns = [_make_single_env for _ in range(num_envs)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    return RecordEpisodeStatistics(envs)


def make_box2d_env(
    name: str,
    num_envs: int = 4,
    max_episode_steps: int = 1000,
    render_mode: str = "rgb_array",
    **kwargs,
) -> VectorEnv:
    """
    Creates a sync vectorized [Box2D](https://box2d.org/) environment.

    Applies wrappers -
    - `gymnasium.wrappers.vector.RecordEpisodeStatistics`

    Parameters
    ----------
    name : str
        Name of the environment (e.g., `BipedalWalker-v3`)
    num_envs : int (optional)
        The number of vectorized environments to make. Default is `4`
    max_episode_steps : int (optional)
        Maximum number of episode steps. Default is `1000`
    render_mode : str (optional)
        The type of render mode for the environment.
        Default is `rgb_array`
    kwargs : Any (optional)
        Additional arguments passed to `gym.make_vec()`

    Returns
    -------
    envs : VectorEnv
        A set of wrapped vectorized environments
    """
    import warnings

    warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

    envs = gym.make_vec(
        name,
        num_envs=num_envs,
        vectorization_mode="sync",
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )
    return RecordEpisodeStatistics(envs)


def make(
    name: str,
    *,
    num_envs: int = 4,
    max_episode_steps: int = 1000,
    render_mode: str = "rgb_array",
    **kwargs,
) -> VectorEnv:
    """
    Create a vectorized Gymnasium environment from a registered canonical ID.

    Mirrors `envrax.make` at the call-site so `velora.gym.make(name)` is
    a drop-in alternative for `envrax.make(name)`. The env must be
    registered first via `velora.gym.register` or
    `velora.gym.register_suite`.

    Parameters
    ----------
    name : str
        Registered canonical environment ID (e.g. `"Ant-v5"`,
        `"dm_control/cartpole-balance-v0"`).
    num_envs : int (optional)
        Number of vectorized environments. Default is `4`.
    max_episode_steps : int (optional)
        Maximum number of episode steps. Default is `1000`.
    render_mode : str (optional)
        Render mode for each underlying environment. Default is `"rgb_array"`.
    kwargs : Any (optional)
        Additional arguments forwarded to the suite's `make_fn`.

    Returns
    -------
    envs : VectorEnv
        A set of wrapped vectorized environments.

    Raises
    ------
    unknown_env : ValueError
        If `name` is not registered.
    """
    from velora.gym.envs import get_spec

    spec = get_spec(name)
    return spec.make_fn(
        name,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        **kwargs,
    )
