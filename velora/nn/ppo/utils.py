# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn


def _quiet_moviepy_import() -> None:
    """
    Imports `moviepy.config` with `stderr` suppressed at the file
    descriptor level.

    On Windows, `moviepy` probes for ImageMagick with `dir` shell
    commands during its config import. When ImageMagick is not
    installed, the child process leaks a `File Not Found` message to
    the console that `subprocess` does not capture. Velora never uses
    ImageMagick (`ffmpeg` renders the videos), so the probe's console
    noise is suppressed by importing the config here, before any
    video wrapper triggers it.
    """
    if "moviepy.config" in sys.modules:
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.dup(2)
    os.dup2(devnull, 2)

    try:
        import moviepy.config  # noqa: F401
    except ImportError:
        pass
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull)


def layer_init(
    layer: nn.Linear,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear:
    """
    Initializes a linear layer in-place with orthogonal weights and
    constant bias.

    Parameters
    ----------
    layer : nn.Linear
        The linear layer to initialize
    std : float (optional)
        The gain (scaling factor) for the orthogonal weights.
        Default is `np.sqrt(2)`
    bias_const : float (optional)
        The constant value to fill the bias with. Default is `0.0`

    Returns
    -------
    layer : nn.Linear
        The initialized linear layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(
    env_id: str,
    num_envs: int = 1,
    *,
    gamma: float = 0.99,
    capture_video: bool = True,
    video_dir: Path | str | None = None,
) -> gym.vector.VectorEnv:
    """
    Creates a set of vectorized environments with the standard PPO
    preprocessing wrappers applied to each one.

    Parameters
    ----------
    env_id : str
        The Gymnasium environment ID (e.g., `HalfCheetah-v4`) or a
        MuJoCo Playground ID (e.g., `playground/humanoid-walk-v0`)
    num_envs : int (optional)
        The number of parallel environments. Default is `1`
    gamma : float (optional)
        The discount factor for reward normalization. Default is `0.99`
    capture_video : bool (optional)
        Whether to record videos of the first environment. For
        `playground/` environments the run's final episode can also be
        captured by scheduling it with `record_last_episode`, written
        when the environment is closed. Default is `True`
    video_dir : Path | str (optional)
        Directory to write recorded videos into (e.g., the run's
        wandb directory `runs/{exp_name}/wandb/run-{id}/videos`).
        When `None`, uses `runs/videos`. Default is `None`

    Returns
    -------
    envs : gym.vector.VectorEnv
        The vectorized environments
    """
    if capture_video:
        _quiet_moviepy_import()

    if env_id.startswith("playground/"):
        from velora.envs import make_playground_env

        return make_playground_env(
            env_id,
            num_envs,
            gamma=gamma,
            capture_video=capture_video,
            video_dir=video_dir,
        )

    if video_dir is None:
        video_dir = Path("runs", "videos")

    def thunk(idx: int):
        def _make() -> gym.Env:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, str(video_dir))
            else:
                env = gym.make(env_id)

            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.clip(np.asarray(obs), -10, 10),
                env.observation_space,
            )
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env,
                lambda reward: float(np.clip(float(reward), -10, 10)),
            )
            return env

        return _make

    return gym.vector.SyncVectorEnv(
        [thunk(i) for i in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the advantages and returns for a rollout using
    Generalized Advantage Estimation (GAE), bootstrapping the final
    step from `last_value`.

    Parameters
    ----------
    rewards : torch.Tensor
        The rewards received from the environments
        `(num_steps, num_envs)`
    values : torch.Tensor
        The critic's state-value estimates `(num_steps, num_envs)`
    dones : torch.Tensor
        The environment completion flags after each step
        `(num_steps, num_envs)`
    last_value : torch.Tensor
        The critic's state-value estimates for the observation
        following the rollout's final step `(num_envs, 1)` or
        `(num_envs,)`
    gamma : float
        The discount factor
    gae_lambda : float
        The lambda for the GAE

    Returns
    -------
    advantages : torch.Tensor
        The GAE advantage estimates `(num_steps, num_envs)`
    returns : torch.Tensor
        The discounted returns, `advantages + values`
        `(num_steps, num_envs)`
    """
    last_value = last_value.flatten()
    advantages = torch.zeros_like(rewards)
    last_gae_lambda = 0.0
    num_steps = rewards.shape[0]

    for t in reversed(range(num_steps)):
        next_non_terminal = 1.0 - dones[t]
        next_values = last_value if t == num_steps - 1 else values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae_lambda = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
        )
        advantages[t] = last_gae_lambda

    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardizes a batch of advantages to zero mean and unit standard
    deviation, stabilizing the policy gradient scale.

    Parameters
    ----------
    advantages : torch.Tensor
        A batch of advantages `(batch_size,)`
    eps : float (optional)
        A small constant added to the standard deviation to prevent
        division by zero when all advantages are equal.
        Default is `1e-8`

    Returns
    -------
    advantages : torch.Tensor
        The normalized advantages `(batch_size,)`
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)
