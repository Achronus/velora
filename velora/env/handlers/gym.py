from pathlib import Path
from typing import Any
import gymnasium as gym
from pydantic import PrivateAttr

from velora.conf.config import EnvConfig, load_config
from velora.enums import RenderMode
from velora.env.handlers import EnvHandler


class GymEnvHandler(EnvHandler):
    """A Gym environment handler for working with [Gymnasium](https://gymnasium.farama.org/) environments."""

    config_filepath: Path | str
    env: gym.Env | None = None
    wrappers: list[gym.Wrapper] = []

    _config = PrivateAttr(None)
    _wrapper_config = PrivateAttr(None)

    @property
    def config(self) -> EnvConfig:
        """The environment config settings."""
        return self._config

    @property
    def wrapper_config(self) -> GymWrapperSettings:
        """The environment wrapper config settings."""
        return self._wrapper_config

    def model_post_init(self, __context: Any) -> None:
        self.env = (
            gym.make(self.config.ENV.NAME, render_mode="rgb_array")
            if self.env is None
            else self.env
        )
        self.config = load_config(self.config_filepath)

        if self.wrappers:
            self.env = self.__apply_wrappers()
            self._wrapper_config = load_config(self.config_filepath, as_dict=True)[
                "wrappers"
            ]

    def __apply_wrappers(self) -> gym.Env:
        """Wraps one or more [gymnasium.Wrappers](https://gymnasium.farama.org/api/wrappers/table/) around a Gym environment."""
        env = self.env

        for wrapper in self.wrappers:
            name = ""
            if name in self.wrapper_config.keys():
                env = wrapper(
                    env=env,
                    **self.wrapper_config[name],
                )
            else:
                env = wrapper(env)

        return env

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
