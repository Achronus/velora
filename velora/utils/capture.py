import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import gymnasium as gym
import torch

from velora.gym.wrap import add_core_env_wrappers

if TYPE_CHECKING:
    from velora.models.base import RLAgent  # pragma: no cover


def record_last_episode(
    agent: "RLAgent",
    env_name: str,
    dirname: str,
    root_path: str | Path | None = None,
) -> None:
    """
    Manually makes a video recording of an agent in an episode.

    Used to record the last episode of training runs when
    `TrainCallback.RecordVideos` is applied.

    Filename format: `<env_name>_final-episode-0.mp4`.

    Parameters:
        agent (RLAgent): an agent to use
        env_name (str): the name of environment to use
        dirname (str): the model directory name. Used inside `checkpoints` directory
            as `checkpoints/<dirname>/videos`
        root_path (str, optional): a root path for the checkpoint directory
    """
    cp_path = Path("checkpoints", dirname, "videos")
    dirpath = Path(root_path, cp_path) if root_path else cp_path

    def trigger(t: int) -> bool:
        return True

    env = gym.make(env_name, render_mode="rgb_array")

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

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state

    env.close()


def evaluate_agent(agent: "RLAgent", env: gym.Env) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the performance of an agent on an environment using a single episode.

    Parameters:
        agent (RLAgent): the agent to evaluate
        env (gym.Env): the Gymnasium environment

    Returns:
        ep_return (torch.Tensor): episodic return.
        ep_length (torch.Tensor): episode length.
    """
    state, _ = env.reset()
    hidden = None

    ep_return = 0
    ep_length = 0

    while True:
        action, hidden = agent.predict(state, hidden, train_mode=False)
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        state = next_state

        if done:
            ep_return = info["episode"]["r"]
            ep_length = info["episode"]["l"]
            break

    return ep_return, ep_length
