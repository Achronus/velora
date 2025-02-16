from functools import reduce
import re
from typing import Callable, Literal, List, Tuple

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch


WrapperType = List[gym.Wrapper | gym.vector.VectorWrapper | Callable]

ContinuousGymName = Literal[
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "LunarLanderContinuous-v3",
    "BipedalWalker-v3",
    "CarRacing-v3",
    "Reacher-v5",
    "Pusher-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
]


def wrap_gym_env(
    env: gym.Env | str | ContinuousGymName,
    wrappers: List[gym.Wrapper | gym.vector.VectorWrapper | Callable],
) -> gym.Env:
    """
    Creates a new [Gymnasium](https://gymnasium.farama.org/) environment with
    one or more [gymnasium.Wrappers](https://gymnasium.farama.org/api/wrappers/table/) wrappers applied.

    Parameters:
        env (gymnasium.Env | str): a name of a Gymnasium environment or the
            environment itself to wrap
        wrappers (List[gym.Wrapper | gym.vector.VectorWrapper | functools.partial]): a list of wrapper classes or partially applied wrapper functions

    Examples:
        A Gymnasium environment with normalization and reward clipping:
        ```python
        from functools import partial

        from gymnasium.wrappers import (
            NormalizeObservation,
            NormalizeReward,
            RecordEpisodeStatistics,
            ClipReward,
        )
        from gymnasium.wrappers.numpy_to_torch import NumpyToTorch

        import torch

        env = wrap_gym_env("InvertedPendulum-v5", [
            partial(NormalizeObservation, epsilon=1e-8),
            partial(NormalizeReward, gamma=0.99, epsilon=1e-8),
            partial(ClipReward, max_reward=10.0),
            RecordEpisodeStatistics,
            partial(NumpyToTorch, device=torch.device("cuda"))
        ])
        ```

    Returns:
        env (gymnasium.Env): The wrapped environment
    """
    if isinstance(env, str):
        env = gym.make(env)

    def apply_wrapper(env: gym.Env, wrapper: WrapperType) -> gym.Env:
        return wrapper(env)

    return reduce(apply_wrapper, wrappers, env)


def make_wrapped_vec_env(
    name: str,
    *,
    n_envs: int = 3,
    epsilon: float = 1e-8,
    max_reward: float = 10.0,
    gamma: float = 0.99,
    device: str = "cpu",
) -> gym.vector.SyncVectorEnv:
    """
    Creates a vectorized [Gymnasium environment](https://gymnasium.farama.org/api/vector/) with predefined wrappers.

    Wrappers included:
    - [NormalizeObservation](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.NormalizeObservation)
    - [NormalizeReward](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.NormalizeReward)
    - [RecordEpisodeStatistics](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.RecordEpisodeStatistics)
    - [ClipReward](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.ClipReward)
    - [NumpyToTorch](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.NumpyToTorch)

    Parameters:
        name (str): the name of the environment
        n_envs (int, optional): the number of parallel environments (Default is `3`)
        epsilon (float, optional): stability parameter for normalization scaling.
            Used in the `NormalizeObservation` and `NormalizeReward` wrappers
            (Default is `1e-8`)
        max_reward (float, optional): the max absolute value for discounted return.
            Used in the `ClipReward` wrapper (Default is `10.0`)
        gamma (float, optional): the discount factor used in `NormalizeReward`
            wrapper. Applies to the exponential moving average (Default is `0.99`)
        device (str, optional): the type of CUDA device to load onto.
            Used in the `NumpyToTorch` wrapper (Default is `cpu`)
    """
    from gymnasium.wrappers.vector import (
        NormalizeObservation,
        NormalizeReward,
        RecordEpisodeStatistics,
        ClipReward,
    )
    from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch

    envs = gym.make_vec(
        name,
        num_envs=n_envs,
        vectorization_mode="sync",
    )
    envs = NormalizeObservation(envs, epsilon=epsilon)
    envs = NormalizeReward(envs, gamma=gamma, epsilon=epsilon)
    envs = ClipReward(envs, max_reward=max_reward)
    envs = RecordEpisodeStatistics(envs)
    envs = NumpyToTorch(envs, device=torch.device(device))
    return envs


def get_obs_shape(obs_space: spaces.Space) -> tuple[int, ...]:
    """
    Gets the shape of the observation space (useful for buffers).

    Args:
        obs_space (gymnasium.spaces.Space): A [Gymnasium](https://gymnasium.farama.org/api/spaces/) environments observation space
    """
    if not isinstance(obs_space, (spaces.Discrete, spaces.Box)):
        raise NotImplementedError(f"'{obs_space}' observation space is not supported")

    if isinstance(obs_space, spaces.Box):
        return obs_space.shape

    return (obs_space.n,)


def get_action_size(action_space: spaces.Space) -> int:
    """
    Gets the action space size.

    Args:
        action_space (gymnasium.spaces.Space): A [Gymnasium](https://gymnasium.farama.org/api/spaces/) environments action space
    """
    if not isinstance(action_space, (spaces.Discrete, spaces.Box)):
        raise NotImplementedError(f"'{action_space}' action space is not supported")

    if isinstance(action_space, spaces.Box):
        return np.prod(action_space.shape).item()

    return action_space.n


def get_action_bounds(action_space: spaces.Box) -> Tuple[float, float]:
    """Returns the actions bounds from a Box action space."""
    return action_space.low.item(), action_space.high.item()


def get_latest_env_names() -> List[str]:
    """Returns a list of the latest gymnasium environment names."""
    env_dict = {}

    for env in gym.envs.registry.keys():
        match = re.match(r"(.+)-v(\d+)", env)
        if match:
            base_name, version = match.groups()
            version = int(version)

            # Keep only the latest version
            if base_name not in env_dict or version > env_dict[base_name]:
                env_dict[base_name] = version

    # Reconstruct the latest version environments
    latest_envs = [f"{name}-v{version}" for name, version in env_dict.items()]
    return latest_envs
