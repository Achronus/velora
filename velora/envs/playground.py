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
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import jax
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from mujoco_playground import registry
from mujoco_playground._src.wrapper_torch import (
    RSLRLBraxWrapper,
    _jax_to_torch,
    _torch_to_jax,
)

from velora.envs.normalize import RunningMeanStd
from velora.utils.nn import set_torch_device


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """
    The default episode trigger for video recording, matching
    Gymnasium's `RecordVideo` schedule.

    Records episodes at perfect cubes (0, 1, 8, 27, ...) until episode
    1000, then every 1000th episode.

    Parameters
    ----------
    episode_id : int
        The (zero-based) episode index

    Returns
    -------
    record : bool
        Whether the episode should be recorded
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id

    return episode_id % 1000 == 0


class PlaygroundVectorEnv(gym.vector.VectorEnv):
    """
    A drop-in Gymnasium vectorized environment over MuJoCo Playground.

    Wraps Playground's `RSLRLBraxWrapper` (JAX to PyTorch DLPack bridge)
    and exposes the standard Gymnasium vector API with PyTorch tensors:
    `step` returns the `(obs, rewards, terminations, truncations, info)`
    5-tuple, and completed episodes are reported through
    `info["final_info"]` using the Gymnasium convention.

    Applies the same preprocessing as the CPU wrapper stack: action
    clipping, running observation normalization with clipping, and
    running reward normalization with clipping. Episode statistics are
    accumulated on raw rewards, before normalization.

    Episodes auto-reset on completion (same-step semantics): the
    observation returned on a done step is the first observation of the
    next episode.

    Parameters
    ----------
    env_name : str
        The Playground registry name (e.g., `HumanoidWalk`)
    num_envs : int (optional)
        The number of parallel environments. Default is `1`
    gamma : float (optional)
        The discount factor for reward normalization. Default is `0.99`
    impl : Literal["jax", "warp"] (optional)
        The MJX physics backend. When `None` uses `jax`. The `warp`
        backend currently requires nightly `mujoco`/`warp-lang` builds
        and fails with released versions. Default is `None`
    episode_length : int (optional)
        Maximum steps per episode. When `None` uses the environments
        default config value. Default is `None`
    action_repeat : int (optional)
        Number of physics steps per action. When `None` uses the
        environments default config value. Default is `None`
    device_rank : int (optional)
        The JAX GPU device index. When `None` uses the JAX default
        device (CPU on platforms without JAX GPU support).
        Default is `None`
    normalize_obs : bool (optional)
        Whether to apply running observation normalization.
        Default is `True`
    normalize_reward : bool (optional)
        Whether to apply running reward normalization. Default is `True`
    clip : float (optional)
        The normalized observation and reward clipping bound.
        Default is `10.0`
    capture_video : bool (optional)
        Whether to record videos of the first environment on the
        `capped_cubic_video_schedule`, written to
        `runs/videos/{run_name}`. Videos are rendered and written on
        a background thread. The run's final episode can also be
        captured by scheduling it with `record_last_episode`, written
        when the environment is closed. Default is `True`
    run_name : str (optional)
        The run name used for the video folder. When `None`, uses
        `{env_name}_{timestamp}`. Default is `None`
    """

    _torch_native = True

    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        *,
        gamma: float = 0.99,
        impl: Literal["jax", "warp"] | None = None,
        episode_length: int | None = None,
        action_repeat: int | None = None,
        device_rank: int | None = None,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        clip: float = 10.0,
        capture_video: bool = True,
        run_name: str | None = None,
    ) -> None:
        config = registry.get_default_config(env_name)

        config.impl = impl if impl is not None else "jax"

        raw_env = registry.load(env_name, config=config)

        episode_length = (
            episode_length if episode_length is not None else config.episode_length
        )  # type: ignore
        action_repeat = (
            action_repeat if action_repeat is not None else config.action_repeat
        )  # type: ignore

        self._wrapped = RSLRLBraxWrapper(
            raw_env,
            num_envs,
            0,
            episode_length,
            action_repeat,
            device_rank=device_rank,
        )

        self.device = set_torch_device()
        self.num_envs = num_envs

        obs_dim = int(self._wrapped.num_obs)
        act_dim = int(self._wrapped.num_actions)

        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), np.float32
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

        self.gamma = gamma
        self.clip = clip
        self.eps = 1e-8

        self._obs_stats = (
            RunningMeanStd((obs_dim,), self.device) if normalize_obs else None
        )
        self._reward_stats = (
            RunningMeanStd((), self.device) if normalize_reward else None
        )

        self._returns = torch.zeros(num_envs, device=self.device)
        self._ep_returns = torch.zeros(num_envs, device=self.device)
        self._ep_lengths = torch.zeros(num_envs, device=self.device)
        self._env_device: torch.device | None = None

        self._nan_resets = 0
        self._raw_env = raw_env
        self._record = capture_video

        if not run_name:
            run_name = f"{env_name}_{int(time.time())}"

        self._video_dir = Path("runs", "videos") / run_name
        self._fps = int(round(1.0 / (float(config.ctrl_dt) * action_repeat)))  # type: ignore
        self._episode_length = int(episode_length)  # type: ignore
        self._episode_id = 0
        self._recording = False
        self._states: list[Any] = []
        self._step_count = 0
        self._final_start: int | None = None
        self._video_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="velora-video",
        )
        self._video_future: Future | None = None

        if self._record and sys.platform == "linux":
            os.environ.setdefault("MUJOCO_GL", "egl")

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self._obs_stats is None:
            return obs

        self._obs_stats.update(obs)
        obs = (obs - self._obs_stats.mean) / torch.sqrt(self._obs_stats.var + self.eps)
        return obs.clamp(-self.clip, self.clip)

    def _normalize_reward(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        if self._reward_stats is None:
            return rewards

        self._returns = self.gamma * self._returns * (~dones).float() + rewards
        self._reward_stats.update(self._returns)

        rewards = rewards / torch.sqrt(self._reward_stats.var + self.eps)
        return rewards.clamp(-self.clip, self.clip)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Resets all environments.

        Parameters
        ----------
        seed : int (optional)
            Random number generator seed. When given, reseeds the
            environments JAX reset keys. Default is `None`
        options : dict[str, Any] (optional)
            Ignored. Present for Gymnasium API compatibility.
            Default is `None`

        Returns
        -------
        obs : torch.Tensor
            The initial observations `(num_envs, obs_dim)`
        info : dict
            An empty dict
        """
        if seed is not None:
            self._wrapped.key_reset = jax.random.split(
                jax.random.PRNGKey(seed),  # type: ignore
                self.num_envs,
            )

        obs = self._wrapped.reset()["state"]
        self._env_device = obs.device

        self._returns.zero_()
        self._ep_returns.zero_()
        self._ep_lengths.zero_()

        self._step_count = 0

        if self._record:
            self._episode_id = 0
            self._recording = capped_cubic_video_schedule(0)
            self._states = [self._env0_state()] if self._should_buffer() else []

        return self._normalize_obs(obs.to(self.device).float()), {}

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Steps all environments with a batch of actions.

        Parameters
        ----------
        actions : torch.Tensor
            A batch of actions `(num_envs, act_dim)`, clipped to the
            action bounds internally

        Returns
        -------
        obs : torch.Tensor
            The next observations `(num_envs, obs_dim)`
        rewards : torch.Tensor
            The rewards received `(num_envs,)`
        terminations : torch.Tensor
            Episode termination flags `(num_envs,)`
        truncations : torch.Tensor
            Episode truncation flags `(num_envs,)`
        info : dict
            Contains `final_info` episode stats when any episode
            finished, using the Gymnasium vector convention:
            `{"episode": {"r", "l"}, "_episode": mask}`
        """
        actions = actions.to(self._env_device).clamp(-1.0, 1.0)
        self._wrapped.env_state = self._wrapped.step_fn(
            self._wrapped.env_state,
            _torch_to_jax(actions),
        )
        state = self._wrapped.env_state

        raw_obs = state.obs["state"] if isinstance(state.obs, dict) else state.obs
        obs = _jax_to_torch(raw_obs).to(self.device).float()
        rewards = _jax_to_torch(state.reward).to(self.device).float().view(-1)
        dones = _jax_to_torch(state.done).to(self.device).bool().view(-1)
        truncations = (
            _jax_to_torch(state.info["truncation"]).to(self.device).bool().view(-1)
        )

        bad = ~(torch.isfinite(obs).all(dim=1) & torch.isfinite(rewards))

        if bad.any():
            obs = torch.where(bad.unsqueeze(1), torch.zeros_like(obs), obs)
            rewards = torch.where(bad, torch.zeros_like(rewards), rewards)
            dones = dones | bad
            self._force_reset(bad)

        terminations = dones & ~truncations

        self._ep_returns += rewards
        self._ep_lengths += 1
        self._step_count += 1

        info: dict = {}

        if dones.any():
            info["final_info"] = {
                "episode": {
                    "r": self._ep_returns.clone(),
                    "l": self._ep_lengths.clone(),
                },
                "_episode": dones.clone(),
            }
            self._ep_returns[dones] = 0.0
            self._ep_lengths[dones] = 0.0

        if self._record:
            self._record_step(bool(dones[0]))

        rewards = self._normalize_reward(rewards, dones)

        return self._normalize_obs(obs), rewards, terminations, truncations, info

    def record_last_episode(self, total_steps: int) -> None:
        """
        Schedules recording of the run's final episode.

        Buffers env `0` states only during the last `episode_length`
        `step` calls before `total_steps`. Episodes cannot exceed
        `episode_length` steps, so the final episode always starts
        inside this window and is fully captured. The video is written
        when the environment is closed.

        Parameters
        ----------
        total_steps : int
            The total number of `step` calls the run will make
        """
        self._final_start = max(total_steps - self._episode_length, 0)

    def close(self) -> None:  # type: ignore
        """
        Closes the environments.

        When video recording is enabled and last-episode buffering was
        scheduled with `record_last_episode`, writes the buffered
        states of the final (possibly unfinished) episode as a video
        before closing.
        """
        if not self.closed and self._record and len(self._states) > 1:
            self._write_video()

        self._await_video()
        self._video_executor.shutdown(wait=True)
        super().close()

    def _force_reset(self, bad: torch.Tensor) -> None:
        import jax.numpy as jnp

        self._nan_resets += int(bad.sum())

        bad_jax = _torch_to_jax(bad.to(self._env_device).float())
        state = self._wrapped.env_state
        self._wrapped.env_state = state.replace(done=jnp.maximum(state.done, bad_jax))  # type: ignore

    def _env0_state(self) -> Any:
        return jax.tree_util.tree_map(lambda x: x[0], self._wrapped.env_state)

    def _should_buffer(self) -> bool:
        return self._recording or (
            self._final_start is not None and self._step_count >= self._final_start
        )

    def _record_step(self, done: bool) -> None:
        if not done:
            if self._should_buffer():
                self._states.append(self._env0_state())
            return

        if self._recording:
            self._write_video()

        self._episode_id += 1
        self._recording = capped_cubic_video_schedule(self._episode_id)
        self._states = [self._env0_state()] if self._should_buffer() else []

    def _await_video(self) -> None:
        if self._video_future is not None:
            self._video_future.result()
            self._video_future = None

    def _write_video(self) -> None:
        self._await_video()

        if not self._record:
            return

        self._video_future = self._video_executor.submit(
            self._render_and_write,
            self._states,
            self._episode_id,
        )

    def _render_and_write(self, states: list[Any], episode_id: int) -> None:
        import mediapy

        try:
            frames = self._raw_env.render(states)
            self._video_dir.mkdir(parents=True, exist_ok=True)
            path = self._video_dir / f"rl-video-episode-{episode_id}.mp4"
            mediapy.write_video(str(path), frames, fps=self._fps)
        except Exception as e:
            self._record = False
            self._recording = False
            print(f"Video recording disabled, rendering failed: {e}")
