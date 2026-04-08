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

import os
from functools import partial

import gymnasium as gym
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from velora.gym.error import MissingPackageError
from velora.gym.wrappers import FrameStackReshape


def _make_silent_env(
    name: str,
    render_mode: str,
    wrappers: list,
    **kwargs,
) -> gym.Env:
    """
    Create a single wrapped environment with C-level stdout/stderr
    suppressed during ROM loading.

    Redirects OS file descriptors 1 and 2 to `/dev/null` around the
    `gym.make` call. This catches ALE's C-level `printf` banner that
    fires on ROM initialization — something Python-level
    `redirect_stdout` cannot suppress.

    Parameters
    ----------
    name : str
        Gymnasium environment ID
    render_mode : str
        Render mode for the environment
    wrappers : list
        Ordered list of wrapper callables to apply
    kwargs : Any
        Additional arguments passed to `gym.make()`

    Returns
    -------
    env : gym.Env
        Wrapped environment
    """
    # Redirect OS-level file descriptors to suppress C-level prints
    fd_out = os.dup(1)
    fd_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    try:
        env = gym.make(name, render_mode=render_mode, **kwargs)

        for wrapper in wrappers:
            env = wrapper(env)
        return env

    finally:
        os.dup2(fd_out, 1)
        os.dup2(fd_err, 2)
        os.close(fd_out)
        os.close(fd_err)
        os.close(devnull)


def make_atari_env(
    name: str,
    num_envs: int = 4,
    max_episode_steps: int = 2000,
    render_mode: str = "rgb_array",
    **kwargs,
) -> VectorEnv:
    """
    Creates a sync vectorized [Atari](https://ale.farama.org/) environment
    using common pre-processing techniques.

    Applies wrappers -
    - `gymnasium.wrappers.AtariPreprocessing`
    - `gymnasium.wrappers.FrameStackObservation`
    - `velora.gym.wrappers.FrameStackReshape`
    - `gymnasium.wrappers.TimeLimit`
    - `gymnasium.wrappers.vector.RecordEpisodeStatistics`

    Parameters
    ----------
    name : str
        Name of the environment (e.g., "ALE/Breakout-v5")
    num_envs : int (optional)
        The number of vectorized environments to make. Default is `4`
    max_episode_steps : int (optional)
        Maximum number of episode steps. Default is `2000`
    render_mode : str (optional)
        The type of render mode for the environment.
        Default is `rgb_array`
    kwargs : Any (optional)
        Additional arguments passed to `gym.make_vec()`

    Returns
    -------
    envs : JaxConversion
        A set of wrapped vectorized environments
    """
    try:
        import ale_py

        gym.register_envs(ale_py)
    except ImportError:
        raise MissingPackageError(
            "Atari environments require 'ale-py'. "
            "Install with: pip install 'velora[discrete]'"
        )

    # Compute effective step limit - use smaller than max where possible
    spec = gym.spec(name)
    natural_limit = spec.max_episode_steps
    effective_limit = (
        min(natural_limit, max_episode_steps)
        if natural_limit is not None
        else max_episode_steps
    )

    preprocess = partial(
        AtariPreprocessing,
        noop_max=10,
        frame_skip=4,
        screen_size=84,  # (84, 84)
        grayscale_obs=True,
        grayscale_newaxis=True,  # (84, 84, 1)
    )
    framestack = partial(FrameStackObservation, stack_size=4)  # (84, 84, 4))
    time_limit = partial(TimeLimit, max_episode_steps=effective_limit)
    wrappers = [preprocess, framestack, FrameStackReshape, time_limit]

    env_fns = [
        partial(
            _make_silent_env,
            name,
            render_mode,
            wrappers,
            frameskip=1,  # Handled by AtariPreprocessing
            **kwargs,
        )
        for _ in range(num_envs)
    ]

    envs = gym.vector.SyncVectorEnv(env_fns)
    return RecordEpisodeStatistics(envs)


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
    try:
        import mujoco  # noqa: F401, # type: ignore
    except ImportError:
        raise MissingPackageError(
            "MuJoCo environments require 'mujoco'. "
            "Install with: pip install 'velora[continuous]'"
        )

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

    Wraps each `dm_control.suite` environment in
    `velora.gym.dm_control.DmControlSuiteEnv` for Gymnasium compatibility,
    then vectorizes with `SyncVectorEnv`.

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
            "Install with: pip install 'velora[dmc]'"
        )

    from velora.gym.dm_control.compat import DmControlSuiteEnv

    # Parse "dm_control/{domain}-{task}-v0" → domain, task
    env_id = name
    if env_id.startswith("dm_control/"):
        env_id = env_id[len("dm_control/"):]
    if env_id.endswith("-v0"):
        env_id = env_id[:-len("-v0")]

    # Split on first hyphen: "ball_in_cup-catch" → ("ball_in_cup", "catch")
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
    try:
        import Box2D  # noqa: F401, # type: ignore
    except ImportError:
        raise MissingPackageError(
            "Box2D environments require 'box2d-py'. "
            "Install with: pip install 'velora[continuous]'"
        )

    import warnings

    # Suppress pkg_resources deprecation warning triggered by pygame
    # (transitive Box2D → CarRacing → pygame → pkg_resources import)
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
