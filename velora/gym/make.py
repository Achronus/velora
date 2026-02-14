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

from functools import partial
from typing import Literal

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from velora.gym.error import MissingPackageError
from velora.gym.wrappers import FrameStackReshape, JaxConversion

VectorMode = Literal["sync", "async", "vector_entry_point"]


def make_atari_env(
    name: str,
    num_envs: int,
    vec_mode: VectorMode = "sync",
    render_mode: str = "rgb_array",
    **kwargs,
) -> JaxConversion:
    """
    Creates a vectorized [Atari](https://ale.farama.org/) environment using common
    pre-processing techniques.

    Applies wrappers -
    - `gymnasium.wrappers.AtariPreprocessing`
    - `gymnasium.wrappers.FrameStackObservation`
    - `velora.gym.wrappers.FrameStackReshape`
    - `gymnasium.wrappers.vector.RecordEpisodeStatistics`
    - `velora.gym.wrappers.JaxConversion`

    Parameters
    ----------
    name : str
        Name of the environment (e.g., "ALE/Breakout-v5")
    num_envs : int
        The number of vectorized environments to make
    vec_mode : Literal["sync", "async", "vector_entry_point"] (optional)
        The type of vector environment to make. Default is `sync`
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
    preprocess = partial(
        AtariPreprocessing,
        noop_max=10,
        frame_skip=4,
        screen_size=84,  # (84, 84)
        grayscale_obs=True,
        grayscale_newaxis=True,  # (84, 84, 1)
    )
    framestack = partial(FrameStackObservation, stack_size=4)  # (84, 84, 4))

    try:
        import ale_py

        gym.register_envs(ale_py)
    except ImportError as e:
        raise MissingPackageError(
            "Atari environments require 'ale-py'. "
            "Install with: pip install 'gymnasium[atari]'"
        ) from e

    envs = gym.make_vec(
        name,
        num_envs=num_envs,
        vectorization_mode=vec_mode,
        render_mode=render_mode,
        wrappers=[preprocess, framestack, FrameStackReshape],
        frameskip=1,  # Handled by AtariPreprocessing
        **kwargs,
    )
    return JaxConversion(RecordEpisodeStatistics(envs))


def make_procgen_env(
    name: str,
    num_envs: int,
    vec_mode: VectorMode = "sync",
    render_mode: str = "rgb_array",
    num_levels: int = 0,
    start_level: int = 0,
    distribution_mode: str = "easy",
    **kwargs,
) -> JaxConversion:
    """
    Creates a vectorized [Procgen](https://github.com/Achronus/procgen-gymnasium) environment.

    Procgen environments output 64x64 RGB images by default.

    Applies wrappers -
    - `gymnasium.wrappers.vector.RecordEpisodeStatistics`
    - `velora.gym.wrappers.JaxConversion`

    Parameters
    ----------
    name : str
        Name of the environment (e.g., "procgen_gym/procgen-coinrun-v0")
    num_envs : int
        The number of vectorized environments to make
    vec_mode : Literal["sync", "async", "vector_entry_point"] (optional)
        The type of vector environment to make. Default is `sync`
    render_mode : str (optional)
        The type of render mode for the environment.
        Default is `rgb_array`
    num_levels : int (optional)
        Number of unique levels to use. 0 means unlimited. Default is `0`
    start_level : int (optional)
        The starting level seed. Default is `0`
    distribution_mode : str (optional)
        The difficulty distribution ("easy", "hard", "extreme", "memory",
        "exploration"). Default is `easy`
    kwargs : Any (optional)
        Additional arguments passed to `gym.make_vec()`

    Returns
    -------
    envs : JaxConversion
        A set of wrapped vectorized environments

    Raises
    ------
    MissingPackageError
        If procgen is not installed
    """
    try:
        import procgen_gym
    except ImportError as e:
        raise MissingPackageError(
            "Procgen environments require 'procgen_gym'. "
            "Install with: pip install procgen_gym"
        ) from e

    envs = gym.make_vec(
        name,
        num_envs=num_envs,
        vectorization_mode=vec_mode,
        render_mode=render_mode,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        **kwargs,
    )
    return JaxConversion(RecordEpisodeStatistics(envs))


def make_dmlab_env(
    name: str,
    num_envs: int,
    vec_mode: VectorMode = "sync",
    render_mode: str | None = None,
    width: int = 84,
    height: int = 84,
    fps: int = 60,
    observations: str = "RGBD",
    renderer: str = "software",
    **kwargs,
) -> JaxConversion:
    """
    Creates a vectorized [DeepMind Lab](https://github.com/Achronus/dmlab-gym) environment
    using [Shimmy](https://shimmy.farama.org/environments/dm_lab/) for Gymnasium compatibility.

    Applies wrappers -
    - `shimmy.dm_lab_compatibility.DmLabCompatibilityV0`
    - `gymnasium.wrappers.vector.RecordEpisodeStatistics`
    - `velora.gym.wrappers.JaxConversion`

    Parameters
    ----------
    name : str
        Name of the level (e.g., "rooms_watermaze")
    num_envs : int
        The number of vectorized environments to make
    vec_mode : Literal["sync", "async", "vector_entry_point"] (optional)
        The type of vector environment to make. Default is `sync`
    render_mode : str (optional)
        The type of render mode for the environment. Choices: `["human", None]`.
        Default is `None`
    width : int (optional)
        Width of the observation. Default is `84`
    height : int (optional)
        Height of the observation. Default is `84`
    fps : int (optional)
        Frames per second. Default is `60`
    observations : str (optional)
        Type of observations ("RGBD", "RGB", etc.). Default is `RGBD`
    renderer : str (optional)
        Renderer to use ("software" or "hardware"). Default is `software`
    kwargs : Any (optional)
        Additional arguments passed to DmLabCompatibilityV0

    Returns
    -------
    envs : JaxConversion
        A set of wrapped vectorized environments

    Raises
    ------
    MissingPackageError
        If `deepmind_lab` or `shimmy` is not installed
    """
    try:
        from shimmy.dm_lab_compatibility import DmLabCompatibilityV0
    except ImportError as e:
        raise MissingPackageError(
            "DMLab environments require 'deepmind_lab' and 'shimmy[dm-lab]'. "
            "Install with the compatibility package: pip install 'dmlab-gym'\n"
            "and build 'deepmind_lab' with 'dmlab-gym build' (Linux only). "
            "See: https://github.com/Achronus/dmlab-gym"
        ) from e

    def make_env():
        return DmLabCompatibilityV0(
            level_name=name,
            observations=observations,
            renderer=renderer,
            width=width,
            height=height,
            fps=fps,
            render_mode=render_mode if render_mode == "human" else None,
            **kwargs,
        )

    if vec_mode == "async":
        envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    else:
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

    return JaxConversion(RecordEpisodeStatistics(envs))
