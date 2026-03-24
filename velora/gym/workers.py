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

import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

MakeFn = Callable[..., "VectorEnv"]


def _worker_process(
    cmd_conn: Connection,
    result_conn: Connection,
    env_specs: List[Tuple[str, int]],
    make_fn_ref: Tuple[str, str],
) -> None:
    """
    Environment worker subprocess main loop.

    Creates and steps environments locally. Never touches the GPU —
    `CUDA_VISIBLE_DEVICES` is cleared before any imports that could
    trigger JAX initialization.

    The `make_fn` is passed as a `(module, qualname)` string pair
    rather than a callable to prevent pickle from importing `velora`
    (and triggering console/GPU initialization) before the worker has
    a chance to silence stdout and hide CUDA.

    Parameters
    ----------
    cmd_conn : Connection
        Receives commands from the main process
    result_conn : Connection
        Sends results back to the main process
    env_specs : List[Tuple[str, int]]
        `(env_name, batch_size)` for each trainer this worker owns
    make_fn_ref : Tuple[str, str]
        `(module_path, function_name)` for lazy import of the
        environment factory function
    """
    # Ensure no GPU access — belt and suspenders with parent's env var
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Import velora after silencing
    import importlib

    module = importlib.import_module(make_fn_ref[0])
    make_fn = getattr(module, make_fn_ref[1])

    # Create environments
    envs: List["VectorEnv"] = []
    for env_name, batch_size in env_specs:
        env: "VectorEnv" = make_fn(env_name, batch_size)
        envs.append(env)

    # Reset all environments and send initial observations
    initial_obs = []
    for env in envs:
        obs, _ = env.reset()
        initial_obs.append(obs)

    result_conn.send(initial_obs)

    # Command loop
    while True:
        msg = cmd_conn.recv()
        cmd = msg[0]

        if cmd == "step":
            actions_list = msg[1]
            results = []

            for env, actions in zip(envs, actions_list):
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                results.append((next_obs, rewards, terminated, truncated))

            result_conn.send(results)

        elif cmd == "reset":
            local_idx: int = msg[1]
            env_name, batch_size = env_specs[local_idx]

            envs[local_idx].close()
            envs[local_idx] = make_fn(env_name, batch_size, vec_mode="sync")

            obs, _ = envs[local_idx].reset()
            result_conn.send(obs)

        elif cmd == "close":
            for env in envs:
                env.close()
            break


class EnvWorkerPool:
    """
    Fixed-size process pool for parallel environment stepping.

    Spawns `num_workers` subprocesses using the `spawn` start method,
    each owning a contiguous slice of trainers' environments. Workers
    create and step environments locally — they never import JAX with
    GPU support or allocate VRAM.

    The pool is created once during setup and reused for the entire
    training run. Each collection timestep fans out actions to all
    workers, then collects `(obs, rewards, terminated, truncated)`
    back through pipes.

    Remainder trainers are distributed one contiguous slice each across
    the first `remainder` workers for even load balancing.

    Parameters
    ----------
    env_specs : List[Tuple[str, MakeFn, int]]
        `(env_name, make_fn, batch_size)` for each trainer
    num_workers : int
        Number of worker processes to spawn. Each worker owns
        `ceil(num_trainers / num_workers)` trainers
    """

    def __init__(
        self,
        env_specs: List[Tuple[str, MakeFn, int]],
        num_workers: int,
    ) -> None:
        self.num_trainers = len(env_specs)
        self.num_workers = min(num_workers, self.num_trainers)

        # All trainers share the same make_fn
        _make_fn = env_specs[0][1]
        self._make_fn_ref = (_make_fn.__module__, _make_fn.__qualname__)

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

        # Spawn workers using "spawn" context — no fork, no JAX
        # thread state inheritance, no deadlock risk
        ctx = mp.get_context("spawn")

        self._cmd_conns: List[Connection] = []
        self._result_conns: List[Connection] = []
        self._processes: List[mp.Process] = []

        # Hide CUDA from child processes and mark as worker subprocess
        _old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["_VELORA_WORKER"] = "1"

        for w in range(self.num_workers):
            s, e = self._worker_slices[w]
            worker_specs = [
                (name, batch_size) for name, _, batch_size in env_specs[s:e]
            ]

            parent_cmd, child_cmd = ctx.Pipe()
            parent_result, child_result = ctx.Pipe()

            p = ctx.Process(
                target=_worker_process,
                args=(child_cmd, child_result, worker_specs, self._make_fn_ref),
                daemon=True,
            )
            p.start()

            # Close child-side connections in parent (avoids fd leak)
            child_cmd.close()
            child_result.close()

            self._cmd_conns.append(parent_cmd)  # type: ignore
            self._result_conns.append(parent_result)  # type: ignore
            self._processes.append(p)  # type: ignore

        # Verify all workers are alive
        dead = [w for w, p in enumerate(self._processes) if not p.is_alive()]
        if dead:
            self.close()
            raise RuntimeError(
                f"Env workers {dead} died on startup. Process/thread limits reached."
                f"Try reducing num_env_workers (currently {num_workers}). "
            )

        # Restore CUDA visibility in parent
        if _old_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _old_cuda
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        os.environ.pop("_VELORA_WORKER", None)

        # Collect initial observations from all workers
        self.initial_obs = self._collect_initial_obs()

        # Pre-allocate reusable buffers for step_all results
        sample_obs: np.ndarray = self.initial_obs[0]  # (B, H, W, C)
        batch_size = sample_obs.shape[0]

        self._step_obs = np.empty(
            (self.num_trainers, *sample_obs.shape),
            dtype=sample_obs.dtype,
        )
        self._step_rewards = np.empty(
            (self.num_trainers, batch_size),
            dtype=np.float32,
        )
        self._step_terminated = np.empty(
            (self.num_trainers, batch_size),
            dtype=np.bool_,
        )
        self._step_truncated = np.empty(
            (self.num_trainers, batch_size),
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
            obs_list = self._result_conns[w].recv()
            all_obs.extend(obs_list)

        return np.stack(all_obs)

    def step_all(
        self,
        all_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all environments in parallel across workers.

        Fans out actions to all workers, then collects results.
        Workers step their assigned trainers sequentially within
        each process, but all workers run simultaneously.

        Returns views into internal buffers — valid until the next
        `step_all` call.

        Parameters
        ----------
        all_actions : np.ndarray
            Actions for all trainers. Shape: `(P, B, 1)` int32

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
        # Fan out — send each worker its slice of actions
        for w in range(self.num_workers):
            s, e = self._worker_slices[w]
            worker_actions = [all_actions[i].squeeze(-1) for i in range(s, e)]
            self._cmd_conns[w].send(("step", worker_actions))

        # Collect — write into pre-allocated buffers
        for w in range(self.num_workers):
            results = self._result_conns[w].recv()
            s, _ = self._worker_slices[w]

            for local_idx, (obs, rew, term, trunc) in enumerate(results):
                global_idx = s + local_idx
                np.copyto(self._step_obs[global_idx], obs)
                np.copyto(self._step_rewards[global_idx], rew)
                np.copyto(self._step_terminated[global_idx], term)
                np.copyto(self._step_truncated[global_idx], trunc)

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
        self._cmd_conns[w].send(("reset", local_idx))
        return self._result_conns[w].recv()

    def close(self) -> None:
        """Shut down all worker processes."""
        for w in range(self.num_workers):
            try:
                self._cmd_conns[w].send(("close",))
            except (BrokenPipeError, OSError):
                pass

        for p in self._processes:
            p.join(timeout=5)

            if p.is_alive():
                p.terminate()
