# Copyright 2025 Achronus
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

import json
import os
import struct
import sys
from abc import ABC, abstractmethod
from typing import List

import gymnasium as gym
import numpy as np

CMD_STEP = 1
CMD_RESET = 2
CMD_CLOSE = 3


class _FrameStackReshape(gym.ObservationWrapper):
    """
    Moves frame stack dimension to channels: `(FS, H, W, 1) → (H, W, FS)`.

    Inlined copy of `velora.gym.wrappers.FrameStackReshape`.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)
        old_shape = old_space.shape
        assert old_shape is not None and len(old_shape) == 4
        FS, H, W, C = old_shape
        assert C == 1, f"Expected grayscale input with C=1, got C={C}"

        self.observation_space = gym.spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=(H, W, FS),
            dtype=old_space.dtype,  # type: ignore
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(np.squeeze(observation, axis=-1), (1, 2, 0))


def _read_exact(fd: int, n: int) -> bytes:
    """Read exactly `n` bytes from a file descriptor."""
    chunks = []
    remaining = n

    while remaining > 0:
        chunk = os.read(fd, remaining)

        if not chunk:
            raise EOFError("Worker pipe closed unexpectedly")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes to a file descriptor."""
    view = memoryview(data)
    offset = 0

    while offset < len(view):
        written = os.write(fd, view[offset:])
        offset += written


def _create_atari_env(
    name: str,
    batch_size: int,
    max_episode_steps: int,
) -> gym.vector.VectorEnv:
    """
    Create a `SyncVectorEnv` with standard Atari preprocessing.

    Inlined from `velora.gym.make.make_atari_env` to avoid importing
    velora (and transitively JAX) in the worker subprocess.
    """
    from gymnasium.vector import SyncVectorEnv
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit

    spec = gym.spec(name)
    natural_limit = spec.max_episode_steps
    effective_limit = (
        min(natural_limit, max_episode_steps)
        if natural_limit is not None
        else max_episode_steps
    )

    def _make_single() -> gym.Env:
        env = gym.make(name, render_mode="rgb_array", frameskip=1)
        env = AtariPreprocessing(
            env,
            noop_max=10,
            frame_skip=4,
            screen_size=84,
            grayscale_obs=True,
            grayscale_newaxis=True,
        )
        env = FrameStackObservation(env, stack_size=4)
        env = _FrameStackReshape(env)
        env = TimeLimit(env, max_episode_steps=effective_limit)
        return env

    return SyncVectorEnv([_make_single for _ in range(batch_size)])


def _create_generic_env(
    name: str,
    batch_size: int,
    max_episode_steps: int,
) -> gym.vector.VectorEnv:
    """
    Create a sync vectorized environment via `gym.make_vec`.

    Works for any standard Gymnasium environment (MuJoCo, Box2D, etc.).
    """
    return gym.make_vec(
        name,
        num_envs=batch_size,
        vectorization_mode="sync",
        max_episode_steps=max_episode_steps,
    )


def _create_dmc_env(
    name: str,
    batch_size: int,
    max_episode_steps: int,
) -> gym.vector.VectorEnv:
    """
    Create a sync vectorized DeepMind Control Suite environment.

    Mirrors `velora.gym.make.make_dmc_env`.
    """
    from velora.gym.dm_control.compat import DmControlSuiteEnv

    import dm_control.suite  # type: ignore
    from gymnasium.wrappers import TimeLimit

    # Parse "dm_control/{domain}-{task}-v0" → domain, task
    env_id = name
    if env_id.startswith("dm_control/"):
        env_id = env_id[len("dm_control/"):]
    if env_id.endswith("-v0"):
        env_id = env_id[: -len("-v0")]

    parts = env_id.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid DMC environment name: {name}")

    domain, task = parts

    def _make_single_env():
        env = dm_control.suite.load(domain_name=domain, task_name=task)
        wrapped = DmControlSuiteEnv(env)
        return TimeLimit(wrapped, max_episode_steps=max_episode_steps)

    env_fns = [_make_single_env for _ in range(batch_size)]
    return gym.vector.SyncVectorEnv(env_fns)


def _create_env(
    name: str,
    batch_size: int,
    max_episode_steps: int,
) -> gym.vector.VectorEnv:
    """Dispatch to the correct environment creator based on env name prefix."""
    if name.startswith("ALE/"):
        return _create_atari_env(name, batch_size, max_episode_steps)
    elif name.startswith("dm_control/"):
        return _create_dmc_env(name, batch_size, max_episode_steps)
    else:
        return _create_generic_env(name, batch_size, max_episode_steps)


class EnvWorker(ABC):
    """
    Base worker that owns the command loop, env lifecycle, and
    observation / reward / flag protocol.

    Subclasses define how actions are sized and parsed.

    Parameters
    ----------
    env_specs : List[List]
        `[name, batch_size]` for each environment
    max_episode_steps : int
        Maximum episode steps for environments
    cmd_fd : int
        File descriptor to read commands from
    result_fd : int
        File descriptor to write results to
    """

    def __init__(
        self,
        env_specs: List[List],
        max_episode_steps: int,
        cmd_fd: int,
        result_fd: int,
    ) -> None:
        self._env_specs = env_specs
        self._max_episode_steps = max_episode_steps
        self._cmd_fd = cmd_fd
        self._result_fd = result_fd

        # Register ALE only when needed
        if any(spec[0].startswith("ALE/") for spec in env_specs):
            import ale_py

            gym.register_envs(ale_py)

        # Create and reset all environments
        self._envs: List[gym.vector.VectorEnv] = []
        self._initial_obs: List[np.ndarray] = []

        for name, batch_size in env_specs:
            env = _create_env(name, batch_size, max_episode_steps)
            obs, _ = env.reset()
            self._envs.append(env)
            self._initial_obs.append(self._process_obs(obs, len(self._envs) - 1))

        self._batch_size: int = self._initial_obs[0].shape[0]
        self._num_envs = len(self._envs)

    @abstractmethod
    def _action_nbytes(self) -> int:
        """Total bytes per environment for one step's worth of actions."""

    @abstractmethod
    def _parse_actions(self, raw_bytes: bytes, env_idx: int) -> np.ndarray:
        """Decode raw action bytes into a numpy array for `env.step`."""

    def _send_handshake(self) -> None:
        """Send obs shape info + initial observations to the parent."""
        sample_obs = self._initial_obs[0]
        shapes = {
            "obs_shape": list(sample_obs.shape),
            "obs_dtype": str(sample_obs.dtype),
        }
        json_bytes = json.dumps(shapes).encode()
        _write_all(self._result_fd, struct.pack("<I", len(json_bytes)))
        _write_all(self._result_fd, json_bytes)

        for obs in self._initial_obs:
            _write_all(self._result_fd, obs.tobytes())

    def _handle_step(self) -> None:
        """Read actions, step all envs, write results."""
        action_nb = self._action_nbytes()
        all_action_bytes = _read_exact(
            self._cmd_fd, self._num_envs * action_nb
        )

        result_parts = []
        for i, env in enumerate(self._envs):
            raw = all_action_bytes[i * action_nb : (i + 1) * action_nb]
            actions = self._parse_actions(raw, i)

            obs, rew, term, trunc, _ = env.step(actions)
            obs = self._process_obs(obs, i)

            result_parts.append(obs.tobytes())
            result_parts.append(np.asarray(rew, dtype=np.float32).tobytes())
            result_parts.append(np.asarray(term, dtype=np.uint8).tobytes())
            result_parts.append(np.asarray(trunc, dtype=np.uint8).tobytes())

        _write_all(self._result_fd, b"".join(result_parts))

    def _handle_reset(self) -> None:
        """Reset a single env by local index."""
        idx_bytes = _read_exact(self._cmd_fd, 4)
        local_idx = struct.unpack("<I", idx_bytes)[0]

        name, bs = self._env_specs[local_idx]
        self._envs[local_idx].close()
        self._envs[local_idx] = _create_env(name, bs, self._max_episode_steps)
        obs, _ = self._envs[local_idx].reset()
        obs = self._process_obs(obs, local_idx)

        self._on_env_reset(local_idx)
        _write_all(self._result_fd, obs.tobytes())

    def _process_obs(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """Hook for subclasses to transform observations (e.g., padding)."""
        return obs

    def _on_env_reset(self, local_idx: int) -> None:
        """Hook for subclasses to update per-env metadata after a reset."""

    def run(self) -> None:
        """Main command loop."""
        self._send_handshake()

        try:
            while True:
                cmd = _read_exact(self._cmd_fd, 1)[0]

                if cmd == CMD_STEP:
                    self._handle_step()
                elif cmd == CMD_RESET:
                    self._handle_reset()
                elif cmd == CMD_CLOSE:
                    break
        finally:
            for env in self._envs:
                try:
                    env.close()
                except Exception:
                    pass


class DiscreteEnvWorker(EnvWorker):
    """`int32` scalar actions — one action per vectorized env instance."""

    def _action_nbytes(self) -> int:
        return self._batch_size * np.dtype(np.int32).itemsize

    def _parse_actions(self, raw_bytes: bytes, env_idx: int) -> np.ndarray:
        return np.frombuffer(raw_bytes, dtype=np.int32)


class ContinuousEnvWorker(EnvWorker):
    """
    `float32` multi-dimensional actions padded to `max_action_dim`.

    The parent sends `(B, max_action_dim)` floats per env. This worker
    trims each env's slice to its actual action dimensionality before
    calling `env.step`.

    When `max_obs_dim > 0`, observations are zero-padded to
    `(B, max_obs_dim)` so all environments report the same obs shape
    to the parent (required for `jax.vmap` across trainers).

    Parameters
    ----------
    max_action_dim : int
        Maximum action dimensionality across all environments
    max_obs_dim : int (optional)
        Maximum observation dimensionality. When non-zero, all obs are
        zero-padded to this width. Default is `0` (no padding)
    """

    def __init__(
        self,
        env_specs: List[List],
        max_episode_steps: int,
        cmd_fd: int,
        result_fd: int,
        max_action_dim: int,
        max_obs_dim: int = 0,
    ) -> None:
        self._max_obs_dim = max_obs_dim
        self._obs_dims: List[int] = []

        super().__init__(env_specs, max_episode_steps, cmd_fd, result_fd)
        self._max_action_dim = max_action_dim

        self._action_dims: List[int] = []
        for env in self._envs:
            space = env.single_action_space
            assert isinstance(space, gym.spaces.Box)
            self._action_dims.append(int(np.prod(space.shape)))

    def _process_obs(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """Normalize to float32 and zero-pad vector observations to `max_obs_dim`."""
        obs = obs.astype(np.float32)

        # Track obs dim on first encounter (during __init__)
        if env_idx >= len(self._obs_dims):
            self._obs_dims.append(obs.shape[-1])

        if self._max_obs_dim <= 0 or obs.shape[-1] == self._max_obs_dim:
            return obs

        padded = np.zeros(
            (*obs.shape[:-1], self._max_obs_dim), dtype=np.float32
        )
        padded[..., : obs.shape[-1]] = obs
        return padded

    def _action_nbytes(self) -> int:
        return (
            self._batch_size * self._max_action_dim * np.dtype(np.float32).itemsize
        )

    def _parse_actions(self, raw_bytes: bytes, env_idx: int) -> np.ndarray:
        actions = np.frombuffer(raw_bytes, dtype=np.float32).reshape(
            self._batch_size, self._max_action_dim
        )
        return actions[:, : self._action_dims[env_idx]]

    def _on_env_reset(self, local_idx: int) -> None:
        space = self._envs[local_idx].single_action_space
        assert isinstance(space, gym.spaces.Box)
        self._action_dims[local_idx] = int(np.prod(space.shape))


def main() -> None:
    """Worker process entry point."""
    config = json.loads(sys.argv[1])

    # On Windows, stdin/stdout default to text mode which translates
    # \n ↔ \r\n and corrupts binary data. Set binary mode before any I/O.
    if sys.platform == "win32":
        import msvcrt

        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    # Communication channels: stdin (commands), stdout (results).
    # Save the real stdout fd for communication, then redirect fd 1
    # to devnull so ALE's C-level ROM-loading prints vanish.
    cmd_fd = sys.stdin.fileno()
    result_fd = os.dup(sys.stdout.fileno())

    # Redirect fd 1 to devnull so any prints from environment
    # creation (ALE ROM banners, MuJoCo init, gymnasium logging, etc.)
    # don't corrupt the binary protocol on the communication pipe.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    env_specs = config["env_specs"]
    max_episode_steps = config.get("max_episode_steps", 2000)
    action_type = config.get("action_type", "discrete")

    common_kwargs = dict(
        env_specs=env_specs,
        max_episode_steps=max_episode_steps,
        cmd_fd=cmd_fd,
        result_fd=result_fd,
    )

    if action_type == "continuous":
        worker = ContinuousEnvWorker(
            **common_kwargs,
            max_action_dim=config["max_action_dim"],
            max_obs_dim=config.get("max_obs_dim", 0),
        )
    else:
        worker = DiscreteEnvWorker(**common_kwargs)

    worker.run()


if __name__ == "__main__":
    main()
