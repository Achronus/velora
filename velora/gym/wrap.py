from functools import reduce
from typing import Callable, Literal, List

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch

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
    one or more [gymnasium.Wrappers](https://gymnasium.farama.org/api/wrappers/table/) applied.

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
            ClipReward,
        )

        env = wrap_gym_env("InvertedPendulum-v5", [
            partial(NormalizeObservation, epsilon=1e-8),
            partial(NormalizeReward, gamma=0.99, epsilon=1e-8),
            partial(ClipReward, max_reward=10.0)
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


def add_core_env_wrappers(env: gym.Env, device: torch.device) -> gym.Env:
    """
    Wraps a [Gymnasium](https://gymnasium.farama.org/) environment with the following (in order) if not already applied:

    - [RecordEpisodeStatistics](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordEpisodeStatistics) - for easily retrieving episode statistics.
    - [NumpyToTorch](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.NumpyToTorch) - for turning environment feedback into `PyTorch` tensors.

    Used in all pre-built algorithms.

    Parameters:
        env (gym.Env): the Gymnasium environment
        device (torch.device): the PyTorch device to perform computations on
    """
    has_stats = False
    has_torch = False

    current_env = env
    while hasattr(current_env, "env"):
        if isinstance(current_env, RecordEpisodeStatistics):
            has_stats = True
        if isinstance(current_env, NumpyToTorch):
            has_torch = True

        current_env = current_env.env

    if not has_stats:
        env = RecordEpisodeStatistics(env)

    if not has_torch:
        env = NumpyToTorch(env, device=device)

    return env
