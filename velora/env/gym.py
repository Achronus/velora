from functools import reduce
from typing import Any, Callable

import gymnasium as gym
from gymnasium.wrappers.vector import (
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    RescaleObservation,
    ClipAction,
    ClipReward,
)
from pydantic import BaseModel, PrivateAttr

from velora.config import EnvironmentSettings
from velora.env.utils import get_action_size, get_obs_shape, is_continuous


def wrap_gym_env(env: gym.Env, wrappers: list[gym.Wrapper, Callable]) -> gym.Env:
    """
    Wraps one or more [gymnasium.Wrappers](https://gymnasium.farama.org/api/wrappers/table/) around a Gym environment.

    Args:
        env (gymnasium.Env): The base gymnasium environment to wrap
        wrappers (list[gym.Wrapper | Callable]): a list of wrapper classes or partially applied wrapper functions (default: None)

    Returns:
        env (gymnasium.Env): The wrapped environment
    """
    if not wrappers:
        return env

    def apply_wrapper(env: gym.Env, wrapper: list[gym.Wrapper, Callable]) -> gym.Env:
        return wrapper(env)

    return reduce(apply_wrapper, wrappers, env)


class GymEnvHandler(BaseModel):
    """
    An environment handler for working with [Gymnasium](https://gymnasium.farama.org/) environments.

    Args:
        config (velora.config.EnvironmentSettings): a EnvironmentSettings model containing environment settings from a YAML file
        wrappers (list[gym.vector.VectorWrapper | Callable], optional): a list of wrapper classes or partially applied wrapper functions (default is None)
    """

    config: EnvironmentSettings
    wrappers: list[gym.vector.VectorWrapper | Callable] = []

    _env: gym.vector.SyncVectorEnv = PrivateAttr(...)
    _continuous: bool = PrivateAttr(default=False)

    @property
    def env(self) -> gym.vector.SyncVectorEnv:
        return self._env

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """The shape of the observation space."""
        return get_obs_shape(self.env.single_observation_space)

    @property
    def n_actions(self) -> int:
        """The number of actions."""
        return get_action_size(self.env.single_action_space)

    @property
    def continuous(self) -> bool:
        """Defines if the environment has a continuous action space."""
        return self._continuous

    def model_post_init(self, __context: Any) -> None:
        self._env = self.make_vec_env()
        self._continuous = is_continuous(self.env.action_space)

    def make_vec_env(self) -> gym.vector.SyncVectorEnv:
        """Creates a vectorized environment."""
        envs = gym.make_vec(
            self.config.name,
            num_envs=self.config.n_envs,
            vectorization_mode="sync",
            wrappers=(
                RecordEpisodeStatistics,
                ClipAction,
            ),
        )
        envs = ClipReward(envs, max_reward=self.config.max_reward)
        envs = RescaleObservation(envs, max_obs=self.config.max_obs)
        envs = NormalizeObservation(envs, epsilon=self.config.epsilon)
        envs = NormalizeReward(
            envs,
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
        )
        return envs
