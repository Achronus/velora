from functools import reduce
from typing import Callable, List, Literal

import gymnasium as gym
import torch

WrapperType = List[gym.Wrapper | gym.vector.VectorWrapper | Callable]


ContinuousGymNames = Literal[
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "LunarLanderContinuous-v3",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
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

DiscreteGymNames = Literal[
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCar-v0",
    "LunarLander-v3",
    "Blackjack-v1",
    "Taxi-v3",
    "CliffWalking-v0",
    "FrozenLake-v1",
    "FrozenLake8x8-v1",
]


def wrap_gym_env(
    env: gym.Env | str | DiscreteGymNames | ContinuousGymNames,
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
        env = gym.make(env, render_mode="rgb_array")

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

    Returns:
        env (gym.Env): an updated Gymnasium environment
    """
    if isinstance(env, gym.Env):
        from gymnasium.wrappers import RecordEpisodeStatistics
        from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
    else:
        raise ValueError(f"{type(env)}' is not supported.")

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
