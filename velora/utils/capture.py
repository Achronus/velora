import warnings
from pathlib import Path

import gymnasium as gym

from velora.gym.wrap import add_core_env_wrappers
from velora.models.base import RLAgent


def record_last_episode(
    agent: RLAgent,
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
            action, hidden = agent.predict(state, hidden)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state

    env.close()
