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

    Mirrors the wrapper chain in `velora.gym.make.make_atari_env`.
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


def main() -> None:
    """Worker process main loop."""
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
    cmd_fd = sys.stdin.fileno()  # 0 — pipe from parent
    result_fd = os.dup(sys.stdout.fileno())  # saved copy of fd 1

    # Silence ALE: redirect fd 1 and 2 to devnull
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)  # stdout → devnull (ALE prints go here)
    os.dup2(devnull, 2)  # stderr → devnull
    os.close(devnull)

    env_specs = config["env_specs"]  # [(name, batch_size), ...]
    max_episode_steps = config.get("max_episode_steps", 2000)

    # Register ALE environments
    import ale_py

    gym.register_envs(ale_py)

    # Create and reset all environments
    envs: List[gym.vector.VectorEnv] = []
    initial_obs_list: List[np.ndarray] = []

    for name, batch_size in env_specs:
        env = _create_atari_env(name, batch_size, max_episode_steps)
        obs, _ = env.reset()
        envs.append(env)
        initial_obs_list.append(obs)

    # Send shape info as length-prefixed JSON
    sample_obs: np.ndarray = initial_obs_list[0]
    shapes = {
        "obs_shape": list(sample_obs.shape),
        "obs_dtype": str(sample_obs.dtype),
    }
    json_bytes = json.dumps(shapes).encode()
    _write_all(result_fd, struct.pack("<I", len(json_bytes)))
    _write_all(result_fd, json_bytes)

    # Send initial observations (one per env)
    for obs in initial_obs_list:
        _write_all(result_fd, obs.tobytes())

    # Pre-compute fixed sizes for the hot loop
    action_nbytes = sample_obs.shape[0] * np.dtype(np.int32).itemsize

    num_envs = len(envs)

    # Command loop
    try:
        while True:
            cmd_byte = _read_exact(cmd_fd, 1)
            cmd = cmd_byte[0]

            if cmd == CMD_STEP:
                # Read all actions in one syscall
                all_action_bytes = _read_exact(cmd_fd, num_envs * action_nbytes)

                # Step all envs sequentially, collect results
                result_parts = []
                for i, env in enumerate(envs):
                    offset = i * action_nbytes
                    actions = np.frombuffer(
                        all_action_bytes[offset : offset + action_nbytes],
                        dtype=np.int32,
                    )
                    obs, rew, term, trunc, _ = env.step(actions)

                    result_parts.append(obs.tobytes())
                    result_parts.append(np.asarray(rew, dtype=np.float32).tobytes())
                    result_parts.append(np.asarray(term, dtype=np.uint8).tobytes())
                    result_parts.append(np.asarray(trunc, dtype=np.uint8).tobytes())

                # Write all results in one syscall
                _write_all(result_fd, b"".join(result_parts))

            elif cmd == CMD_RESET:
                idx_bytes = _read_exact(cmd_fd, 4)
                local_idx = struct.unpack("<I", idx_bytes)[0]

                env_name, bs = env_specs[local_idx]
                envs[local_idx].close()
                envs[local_idx] = _create_atari_env(env_name, bs, max_episode_steps)
                obs, _ = envs[local_idx].reset()
                _write_all(result_fd, obs.tobytes())

            elif cmd == CMD_CLOSE:
                break

    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
