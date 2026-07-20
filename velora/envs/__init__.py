# Copyright 2026 Achronus
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


import re

from velora.envs.normalize import RunningMeanStd
from velora.envs.playground import PlaygroundVectorEnv

PLAYGROUND_PREFIX = "playground/"

NAME_ALIASES = {
    "ball_in_cup-catch": "BallInCup",
    "point_mass-easy": "PointMass",
}


def to_registry_name(env_id: str) -> str:
    """
    Converts a `dm_control` style environment ID into a MuJoCo
    Playground registry name.

    Parameters
    ----------
    env_id : str
        The prefixed environment ID (e.g.,
        `playground/humanoid-walk-v0`). Registry names (e.g.,
        `HumanoidWalk`) are also accepted

    Returns
    -------
    name : str
        The Playground registry name (e.g., `HumanoidWalk`)

    Raises
    ------
    unknown_env : ValueError
        Error when the ID does not match a registered environment
    """
    from mujoco_playground import registry

    name = re.sub(r"-v\d+$", "", env_id.removeprefix(PLAYGROUND_PREFIX))

    if name in registry.ALL_ENVS:
        return name

    if name in NAME_ALIASES:
        return NAME_ALIASES[name]

    converted = "".join(
        part.capitalize() for part in name.replace("-", "_").split("_")
    )

    if converted in registry.ALL_ENVS:
        return converted

    raise ValueError(
        f"Unknown environment: '{env_id}'. "
        "Expected a `dm_control` style ID (e.g., "
        "'playground/humanoid-walk-v0') or a Playground registry name."
    )


def make_playground_env(
    env_id: str,
    num_envs: int = 1,
    *,
    gamma: float = 0.99,
    capture_video: bool = True,
    run_name: str | None = None,
    **env_kwargs,
) -> PlaygroundVectorEnv:
    """
    Creates a vectorized MuJoCo Playground environment.

    Parameters
    ----------
    env_id : str
        The prefixed environment ID in `dm_control` style (e.g.,
        `playground/humanoid-walk-v0`)
    num_envs : int (optional)
        The number of parallel environments. Default is `1`
    gamma : float (optional)
        The discount factor for reward normalization. Default is `0.99`
    capture_video : bool (optional)
        Whether to record videos of the first environment. The run's
        final episode can also be captured by scheduling it with
        `record_last_episode`, written when the environment is closed.
        Default is `True`
    run_name : str (optional)
        The run name used for the video folder. When `None`, uses
        `{env_name}_{timestamp}`. Default is `None`
    **env_kwargs
        Additional `PlaygroundVectorEnv` arguments (e.g., `impl`,
        `device_rank`, `normalize_obs`)

    Returns
    -------
    envs : PlaygroundVectorEnv
        The vectorized environments
    """
    return PlaygroundVectorEnv(
        to_registry_name(env_id),
        num_envs,
        gamma=gamma,
        capture_video=capture_video,
        run_name=run_name,
        **env_kwargs,
    )


__all__ = [
    "PlaygroundVectorEnv",
    "RunningMeanStd",
    "make_playground_env",
    "to_registry_name",
]