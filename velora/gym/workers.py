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

import importlib.resources
import json
import os
import struct
import subprocess
import sys
from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

MakeFn = Callable[..., "VectorEnv"]

CMD_STEP = 1
CMD_RESET = 2
CMD_CLOSE = 3


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


def _worker_script_path() -> str:
    """
    Locate `velora/gym/worker.py` on disk.

    Uses `importlib.resources` to handle both source checkouts and
    installed packages (wheels, editable installs).

    Returns
    -------
    path : str
        Absolute path to the worker script
    """
    return str(importlib.resources.files("velora.gym").joinpath("worker.py"))


class EnvWorkerPool:
    """
    Fixed-size process pool for parallel environment stepping.

    Launches `num_workers` subprocesses via `subprocess.Popen`,
    each owning a contiguous slice of trainers' environments.
    Workers are standalone Python scripts that never import
    velora or JAX — no fork, no CUDA context, no GIL contention.

    Communication uses dedicated pipe file descriptors with a binary
    protocol. Each collection timestep fans out actions to all workers
    simultaneously, then collects results.

    Remainder trainers are distributed one contiguous slice each across
    the first `remainder` workers for even load balancing.

    Parameters
    ----------
    env_specs : List[Tuple[str, MakeFn, int]]
        `(env_name, make_fn, batch_size)` for each trainer
    num_workers : int
        Number of worker processes to spawn. Each worker owns
        `ceil(num_trainers / num_workers)` trainers
    max_episode_steps : int (optional)
        Maximum episode steps for environments. Default is `2000`
    action_type : str (optional)
        Action space type: `"discrete"` or `"continuous"`.
        Default is `"discrete"`
    max_action_dim : int (optional)
        Maximum action dimensionality across all environments.
        Only used when `action_type="continuous"`. Default is `1`
    max_obs_dim : int (optional)
        Maximum observation dimensionality across all environments.
        When non-zero, workers zero-pad vector observations to this
        size. Default is `0` (no padding)
    """

    @staticmethod
    def verify_workers(num_workers: int) -> None:
        """
        Verify the system can support the requested number of worker processes.

        Uses `os.cpu_count()` as a cross-platform upper bound — spawning
        more workers than available CPUs wastes resources and risks
        hitting OS thread/process limits inside containers.

        Parameters
        ----------
        num_workers : int
            Requested number of worker processes

        Raises
        ------
        RuntimeError
            If the system cannot support the requested worker count
        """
        cpu_count = os.cpu_count() or 1

        # Reserve 1 core for the main process (JAX, training loop),
        # round down to nearest multiple of 8
        max_workers = max(1, (cpu_count - 1) // 8 * 8)

        if num_workers > max_workers:
            raise RuntimeError(
                f"Requested {num_workers} env workers but only {cpu_count} "
                f"CPU cores detected. "
                f"We recommend 'num_env_workers={max_workers}' instead."
            )

    def __init__(
        self,
        env_specs: List[Tuple[str, MakeFn, int]],
        num_workers: int,
        max_episode_steps: int = 2000,
        action_type: str = "discrete",
        max_action_dim: int = 1,
        max_obs_dim: int = 0,
    ) -> None:
        self.num_trainers = len(env_specs)
        self.num_workers = min(num_workers, self.num_trainers)
        self._action_type = action_type
        self._max_action_dim = max_action_dim

        # Assign trainers to workers in contiguous slices
        self._worker_slices: List[Tuple[int, int]] = []  # (start, end)
        base = self.num_trainers // self.num_workers
        remainder = self.num_trainers % self.num_workers

        start = 0
        for w in range(self.num_workers):
            count = base + (1 if w < remainder else 0)
            self._worker_slices.append((start, start + count))
            start += count

        # Reverse mapping: global trainer idx → (worker_id, local_idx)
        self._trainer_to_worker: List[Tuple[int, int]] = [(0, 0)] * self.num_trainers
        for w, (s, e) in enumerate(self._worker_slices):
            for local_idx, global_idx in enumerate(range(s, e)):
                self._trainer_to_worker[global_idx] = (w, local_idx)

        # Locate worker script — invoked as a file
        worker_script = _worker_script_path()

        # Spawn workers
        self._cmd_fds: List[int] = []
        self._result_fds: List[int] = []
        self._processes: List[subprocess.Popen] = []

        for w in range(self.num_workers):
            s, e = self._worker_slices[w]
            worker_env_specs = [
                (name, batch_size) for name, _, batch_size in env_specs[s:e]
            ]

            config = json.dumps(
                {
                    "env_specs": worker_env_specs,
                    "max_episode_steps": max_episode_steps,
                    "action_type": action_type,
                    "max_action_dim": max_action_dim,
                    "max_obs_dim": max_obs_dim,
                }
            )

            proc = subprocess.Popen(
                [sys.executable, worker_script, config],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            # Get raw fds from the pipe file objects for binary I/O
            self._cmd_fds.append(proc.stdin.fileno())  # type: ignore
            self._result_fds.append(proc.stdout.fileno())  # type: ignore
            self._processes.append(proc)

        # Read shape info from each worker
        shapes = None
        for w in range(self.num_workers):
            json_len = struct.unpack("<I", _read_exact(self._result_fds[w], 4))[0]
            worker_shapes = json.loads(
                _read_exact(self._result_fds[w], json_len).decode()
            )

            if shapes is None:
                shapes = worker_shapes

        assert shapes is not None
        self._obs_shape = tuple(shapes["obs_shape"])
        self._obs_dtype = np.dtype(shapes["obs_dtype"])
        self._batch_size = self._obs_shape[0]

        # Pre-compute fixed sizes for hot loop
        self._obs_nbytes = int(np.prod(self._obs_shape)) * self._obs_dtype.itemsize

        if action_type == "continuous":
            self._action_nbytes = (
                self._batch_size * max_action_dim * np.dtype(np.float32).itemsize
            )
        else:
            self._action_nbytes = self._batch_size * np.dtype(np.int32).itemsize

        self._reward_nbytes = self._batch_size * np.dtype(np.float32).itemsize
        self._flag_nbytes = self._batch_size  # uint8

        self._result_nbytes_per_env = (
            self._obs_nbytes + self._reward_nbytes + self._flag_nbytes * 2
        )

        # Collect initial observations from all workers
        self.initial_obs = self._collect_initial_obs()

        # Pre-allocate reusable buffers for step_all results
        self._step_obs = np.empty(
            (self.num_trainers, *self._obs_shape),
            dtype=self._obs_dtype,
        )
        self._step_rewards = np.empty(
            (self.num_trainers, self._batch_size),
            dtype=np.float32,
        )
        self._step_terminated = np.empty(
            (self.num_trainers, self._batch_size),
            dtype=np.bool_,
        )
        self._step_truncated = np.empty(
            (self.num_trainers, self._batch_size),
            dtype=np.bool_,
        )

    def _collect_initial_obs(self) -> np.ndarray:
        """
        Collect and stack initial observations from all workers.

        Returns
        -------
        obs : np.ndarray
            Stacked initial observations. Shape: `(P, B, ...)`
        """
        all_obs = []
        for w in range(self.num_workers):
            s, e = self._worker_slices[w]

            for _ in range(e - s):
                obs_bytes = _read_exact(self._result_fds[w], self._obs_nbytes)
                obs = np.frombuffer(obs_bytes, dtype=self._obs_dtype).reshape(
                    self._obs_shape
                )
                all_obs.append(obs.copy())

        return np.stack(all_obs)

    def step_all(
        self,
        all_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all environments in parallel across workers.

        Fans out actions to all workers, then collects results.
        Workers step their assigned trainers sequentially within
        each process, but all workers run in parallel.

        Returns views into internal buffers — valid until the next
        `step_all` call.

        Parameters
        ----------
        all_actions : np.ndarray
            Actions for all trainers.
            Discrete: shape `(P, B, 1)` int32.
            Continuous: shape `(P, B, max_action_dim)` float32.

        Returns
        -------
        next_obs : np.ndarray
            Next observations. Shape: `(P, B, H, W, C)`
        rewards : np.ndarray
            Rewards. Shape: `(P, B)`
        terminated : np.ndarray
            Termination flags. Shape: `(P, B)`
        truncated : np.ndarray
            Truncation flags. Shape: `(P, B)`
        """
        # Fan out — send CMD_STEP + actions to each worker
        is_continuous = self._action_type == "continuous"
        for w in range(self.num_workers):
            s, e = self._worker_slices[w]

            if is_continuous:
                action_parts = [
                    all_actions[i].astype(np.float32).tobytes() for i in range(s, e)
                ]
            else:
                action_parts = [
                    all_actions[i].squeeze(-1).astype(np.int32).tobytes()
                    for i in range(s, e)
                ]

            msg = bytes([CMD_STEP]) + b"".join(action_parts)
            _write_all(self._cmd_fds[w], msg)

        # Collect results from each worker
        for w in range(self.num_workers):
            s, e = self._worker_slices[w]
            num_envs = e - s

            result_bytes = _read_exact(
                self._result_fds[w],
                num_envs * self._result_nbytes_per_env,
            )

            offset = 0
            for local_idx in range(num_envs):
                global_idx = s + local_idx

                # Obs
                end = offset + self._obs_nbytes
                obs = np.frombuffer(
                    result_bytes[offset:end], dtype=self._obs_dtype
                ).reshape(self._obs_shape)
                np.copyto(self._step_obs[global_idx], obs)
                offset = end

                # Rewards
                end = offset + self._reward_nbytes
                rew = np.frombuffer(result_bytes[offset:end], dtype=np.float32)
                np.copyto(self._step_rewards[global_idx], rew)
                offset = end

                # Terminated
                end = offset + self._flag_nbytes
                term = np.frombuffer(result_bytes[offset:end], dtype=np.uint8).astype(
                    np.bool_
                )
                np.copyto(self._step_terminated[global_idx], term)
                offset = end

                # Truncated
                end = offset + self._flag_nbytes
                trunc = np.frombuffer(result_bytes[offset:end], dtype=np.uint8).astype(
                    np.bool_
                )
                np.copyto(self._step_truncated[global_idx], trunc)
                offset = end

        return (
            self._step_obs,
            self._step_rewards,
            self._step_terminated,
            self._step_truncated,
        )

    def reset_trainer(self, global_idx: int) -> np.ndarray:
        """
        Reset a trainer's environment in its owning worker.

        The worker closes the old environment, recreates it from the
        original spec, resets it, and returns the new initial
        observation.

        Parameters
        ----------
        global_idx : int
            Global trainer index

        Returns
        -------
        obs : np.ndarray
            Initial observation from the reset. Shape: `(B, ...)`
        """
        w, local_idx = self._trainer_to_worker[global_idx]

        msg = bytes([CMD_RESET]) + struct.pack("<I", local_idx)
        _write_all(self._cmd_fds[w], msg)

        obs_bytes = _read_exact(self._result_fds[w], self._obs_nbytes)
        return (
            np.frombuffer(obs_bytes, dtype=self._obs_dtype)
            .reshape(self._obs_shape)
            .copy()
        )

    def close(self) -> None:
        """Shut down all worker processes."""
        for w in range(self.num_workers):
            try:
                _write_all(self._cmd_fds[w], bytes([CMD_CLOSE]))
            except (BrokenPipeError, OSError):
                pass

        for proc in self._processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()

            # Close pipe file objects (owns the fds)
            if proc.stdin:
                proc.stdin.close()
            if proc.stdout:
                proc.stdout.close()
