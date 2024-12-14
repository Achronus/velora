from functools import reduce
from typing import Callable

import gymnasium as gym
from gymnasium.wrappers.vector import (
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    ClipReward,
)
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch

import torch

from velora.config import EnvironmentSettings
from velora.env.utils import get_action_size, get_obs_shape, is_continuous


WrapperType = list[gym.Wrapper | gym.vector.VectorWrapper | Callable]


def wrap_gym_env(env: gym.Env, wrappers: WrapperType) -> gym.Env:
    """
    Wraps one or more [gymnasium.Wrappers](https://gymnasium.farama.org/api/wrappers/table/) around a Gym environment.

    Args:
        env (gymnasium.Env): The base gymnasium environment to wrap
        wrappers (list[gym.Wrapper | gym.vector.VectorWrapper | Callable]): a list of wrapper classes or partially applied wrapper functions (default: None)

    Returns:
        env (gymnasium.Env): The wrapped environment
    """
    if not wrappers:
        return env

    def apply_wrapper(env: gym.Env, wrapper: WrapperType) -> gym.Env:
        return wrapper(env)

    return reduce(apply_wrapper, wrappers, env)


class GymEnv:
    """
    An environment handler for working with [Gymnasium](https://gymnasium.farama.org/) environments.

    Args:
        config (velora.config.EnvironmentSettings): a EnvironmentSettings model containing environment settings from a YAML file
        gamma (float): the discount factor used in the exponential moving average in the [NormalizeReward](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.NormalizeReward) wrapper
        wrappers (list[gym.Wrapper | gym.vector.VectorWrapper | Callable], optional): a list of wrapper classes or partially applied wrapper functions (default is None)
    """

    def __init__(
        self,
        config: EnvironmentSettings,
        gamma: float,
        device: torch.device,
        wrappers: WrapperType = [],
    ) -> None:
        self.config = config
        self.gamma = gamma
        self.device = device
        self.wrappers = wrappers

        self.env = self.make_vec_env()
        self.n_envs = self.config.n_envs

        self.is_continuous = is_continuous(self.env.single_action_space)
        self.obs_shape = get_obs_shape(self.env.single_observation_space)
        self.n_actions = get_action_size(self.env.single_action_space)

    def make_vec_env(self) -> gym.vector.SyncVectorEnv:
        """Creates a vectorized environment."""
        envs = gym.make_vec(
            self.config.name,
            num_envs=self.config.n_envs,
            vectorization_mode="sync",
        )
        envs = NormalizeObservation(envs, epsilon=self.config.epsilon)
        envs = NormalizeReward(
            envs,
            gamma=self.gamma,
            epsilon=self.config.epsilon,
        )
        envs = ClipReward(envs, max_reward=self.config.max_reward)
        envs = RecordEpisodeStatistics(envs)
        envs = NumpyToTorch(envs, device=self.device)
        return envs
