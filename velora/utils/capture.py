import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import gymnasium as gym
import torch

from velora.gym.wrap import add_core_env_wrappers

if TYPE_CHECKING:
    from velora.models.base import RLModuleAgent  # pragma: no cover


def record_episode(agent: "RLModuleAgent", dirpath: str | Path | None = None) -> None:
    """
    Manually makes a video recording of an agent in an episode.

    Parameters:
        agent (RLModuleAgent): an agent to use
        dirpath (str, optional): a path to save the video. When `None` uses the
            current working directory
    """
    dirpath = Path(dirpath) if dirpath else Path().cwd()

    def trigger(t: int) -> bool:
        return True

    env = gym.make(agent.env.spec.id, render_mode="rgb_array")

    # Ignore folder warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        env = gym.wrappers.RecordVideo(
            env,
            dirpath,
            episode_trigger=trigger,
            name_prefix=f"{env.spec.name}_final",
        )

    env = add_core_env_wrappers(env, device=agent.device)

    for _ in range(1):
        hidden = None
        done = False

        state, _ = env.reset()

        while not done:
            action, hidden = agent.predict(state, hidden, train_mode=False)

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state

    env.close()


def evaluate_agent(agent: "RLModuleAgent") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the performance of an agent on an environment using a single episode.

    Parameters:
        agent (RLModuleAgent): the agent to evaluate

    Returns:
        ep_return (torch.Tensor): episodic return.
        ep_length (torch.Tensor): episode length.
    """
    env = agent.eval_env

    state, _ = env.reset()
    hidden = None

    ep_return = 0
    ep_length = 0

    while True:
        action, hidden = agent.predict(state, hidden, train_mode=False)
        next_state, reward, terminated, truncated, info = env.step(action.flatten())

        done = terminated or truncated
        state = next_state

        if done:
            ep_return = info["episode"]["r"]
            ep_length = info["episode"]["l"]
            break

    return ep_return, ep_length
