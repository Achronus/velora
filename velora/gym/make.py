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
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from velora.gym.error import MissingPackageError
from velora.gym.wrappers import FrameStackReshape

VectorMode = Literal["sync", "async", "vector_entry_point"]


def make_atari_env(
    name: str,
    num_envs: int = 4,
    max_episode_steps: int = 2000,
    vec_mode: VectorMode = "sync",
    render_mode: str = "rgb_array",
    **kwargs,
) -> VectorEnv:
    """
    Creates a vectorized [Atari](https://ale.farama.org/) environment using common
    pre-processing techniques.

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
    try:
        import ale_py

        gym.register_envs(ale_py)
    except ImportError as e:
        raise MissingPackageError(
            "Atari environments require 'ale-py'. "
            "Install with: pip install 'gymnasium[atari]'"
        ) from e

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

    envs = gym.make_vec(
        name,
        num_envs=num_envs,
        vectorization_mode=vec_mode,
        render_mode=render_mode,
        wrappers=[preprocess, framestack, FrameStackReshape, time_limit],
        frameskip=1,  # Handled by AtariPreprocessing
        **kwargs,
    )
    return RecordEpisodeStatistics(envs)
