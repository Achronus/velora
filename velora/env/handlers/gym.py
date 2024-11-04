from typing import Any
import gymnasium as gym

from velora.conf.config import EnvConfig
from velora.enums import RenderMode
from handlers import EnvHandler


class GymEnvHandler(EnvHandler):
    """A Gym environment handler for working with [Gymnasium](https://gymnasium.farama.org/) environments."""

    config: EnvConfig
    env: gym.Env | None = None

    def model_post_init(self, __context: Any) -> None:
        self.env = (
            gym.make(self.config.ENV.NAME, render_mode="rgb_array")
            if self.env is None
            else self.env
        )

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
