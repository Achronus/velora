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
    Creates a vectorized Atari environment using common
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
        Name of the environment
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


def get_n_actions(env_name: str) -> int:
    """
    Gets the number of actions for a Gymnasium environment.

    Parameters
    ----------
    env_name : str
        Name of the Gymnasium environment.

    Returns
    -------
    n_actions : int
        Number of actions in the environment.

    Raises
    ------
    invalid_env : ValueError
        Unsupported environment if it does not have a `Discrete` action space.
    """
    env = gym.make(env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        return env.action_space.n.item()

    raise ValueError(
        f"Unsupported environment. Must have a 'Discrete' action space. Got: {type(env.action_space)}"
    )
