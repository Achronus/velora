from functools import reduce
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from pydantic import PrivateAttr

from velora.config import Config, load_config
from velora.enums import RenderMode
from velora.env import EnvHandler


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


class GymEnvHandler(EnvHandler):
    """
    A Gym environment handler for working with [Gymnasium](https://gymnasium.farama.org/) environments.

    Args:
        config_filepath (Path | str): the filepath to the YAML config file
        env (gymnasium.Env | None): a custom gymnasium environment (default: None)
        wrappers (list[gym.Wrapper | Callable]): a list of wrapper classes or partially applied wrapper functions (default: None)
    """

    config_filepath: Path | str
    env: gym.Env | None = None
    wrappers: list[gym.Wrapper | Callable] = []

    _config = PrivateAttr(None)

    @property
    def config(self) -> Config:
        """The environment config settings."""
        return self._config

    def model_post_init(self, __context: Any) -> None:
        self._config = load_config(self.config_filepath)
        self.env = (
            gym.make(self.config.ENV.NAME, render_mode="rgb_array")
            if self.env is None
            else self.env
        )

        if self.wrappers:
            self.env = wrap_gym_env(self.env, self.wrappers)

    def run_demo(
        self, episodes: int = 10, render_mode: RenderMode | None = RenderMode.HUMAN
    ) -> None:
        env = gym.make(self.config.ENV.NAME, render_mode=render_mode)
        self.__training_loop(env, episodes)

    def __training_loop(
        self, env: gym.Env, episodes: int, seed: int | None = None
    ) -> None:
        """A helper method for creating the training loop."""
        try:
            for i_episode in range(1, episodes + 1):
                state, info = env.reset(seed=seed)
                episode_over = False
                score = 0

                while not episode_over:
                    action = env.action_space.sample()
                    next_state, reward, terminated, truncated, info = env.step(action)
                    score += reward

                    episode_over = terminated or truncated

                print(f"Episode {i_episode}/{episodes}: {score} score")
        finally:
            env.close()
