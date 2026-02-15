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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, List

import jax
import numpy as np

from velora.gym.envs import EnvSpec

if TYPE_CHECKING:
    from velora.disco.config.state import AgentSnapshot
    from velora.disco.train import AgentTrainer


@dataclass(frozen=True)
class SchedulerSettings:
    """
    Configuration for the `EnvironmentScheduler`.

    Parameters
    ----------
    batch_size : int (optional)
        Number of environments to process per meta-step. Environments
        are popped from a shuffled epoch queue in groups of this size.
        Default is `32`
    max_concurrent : int (optional)
        Maximum number of environment processes alive simultaneously
        during a single batch. Controls peak memory usage. Each
        environment within a sub-batch is processed sequentially, so
        this limits how many are held before teardown.
        Default is `8`
    """

    batch_size: int = 32
    max_concurrent: int = 8


class EnvironmentScheduler:
    """
    Manages the lifecycle of `N` environments for memory-efficient meta-training.

    Only `batch_size` environments are alive at any time. Between activations, agent states are serialized to CPU memory and environment processes are terminated.

    Parameters
    ----------
    env_specs : List[EnvSpec]
        Specifications for all environments
    config : SchedulerConfig
        Scheduler configuration
    seed : int
        Random number generator seed
    """

    def __init__(
        self,
        env_specs: List[EnvSpec],
        config: SchedulerSettings,
        *,
        seed: int,
    ) -> None:
        self.env_specs = env_specs
        self.config = config

        self._rng = np.random.default_rng(seed)
        self._num_envs = len(env_specs)

        # Agent states (None = fresh/uninitialized)
        self._snapshots: Dict[int, AgentSnapshot | None] = {
            i: None for i in range(self._num_envs)
        }

        # Index envs by category for stratified shuffling
        self._category_indices: Dict[str, List[int]] = {}
        for i, spec in enumerate(self.env_specs):
            self._category_indices.setdefault(spec.category, []).append(i)

        # Precompute category arrays
        self._all_indices = np.arange(self._num_envs)
        self._cat_ids = np.empty(self._num_envs, dtype=np.intp)

        for cat_label, (cat, indices) in enumerate(self._category_indices.items()):
            for idx in indices:
                self._cat_ids[idx] = cat_label

        # Epoch state
        self._epoch_queue = np.empty(0, dtype=np.intp)
        self._epoch_cursor = 0
        self._epoch_number = 0

        # Init epoch
        self._start_new_epoch()

    @property
    def num_envs(self) -> int:
        """Total number of environments managed by scheduler."""
        return self._num_envs

    @property
    def epoch(self) -> int:
        """Current epoch number (increments each time all envs are visited)."""
        return self._epoch_number

    @property
    def remaining_epochs(self) -> int:
        """Number of environments remaining in current epoch."""
        return len(self._epoch_queue) - self._epoch_cursor

    def _start_new_epoch(self) -> None:
        """
        Begin a new epoch by building a shuffled, category-interleaved queue.

        Each category's indices are shuffled independently, then interleaved round-robin. Consecutive pops from the queue naturally contain a mix of environment categories.

        For example, with 57 Atari, 16 ProcGen, and 30 DMLab envs -
            [A₁, P₁, D₁, A₂, P₂, D₂, ..., A₁₆, P₁₆, D₁₆, A₁₇, D₁₇, ..., A₃₀, D₃₀, A₃₁, ...]

        Each category's internal order is independently shuffled per epoch.
        """
        # Assign shuffled within-category ranks
        ranks = np.empty(self._num_envs, dtype=np.intp)

        for indices in self._category_indices.values():
            cat_indices = np.array(indices)
            shuffled_ranks = np.arange(len(cat_indices))
            self._rng.shuffle(shuffled_ranks)
            ranks[cat_indices] = shuffled_ranks

        # Sort by (rank, cat_id) -> interleaved order
        order = np.lexsort((self._cat_ids, ranks))

        self._epoch_queue = self._all_indices[order]
        self._epoch_cursor = 0
        self._epoch_number += 1

    def next_batch(self) -> List[int]:
        """
        Pop next batch of `config.batch_size` environment indices from epoch queue.

        New epochs are freshly shuffled before popping.

        Returns
        -------
        batch : List[int]
            Environment indices for this meta-step. Length: `min(batch_size, remaining_epochs)`
        """
        if self._epoch_cursor >= len(self._epoch_queue):
            self._start_new_epoch()

        start = self._epoch_cursor
        end = min(start + self.config.batch_size, len(self._epoch_queue))
        batch = self._epoch_queue[start:end]
        self._epoch_cursor = end

        return batch.tolist()

    def iterate_concurrent(self, batch: List[int]) -> Iterator[List[int]]:
        """
        Split a batch into sub-batches matching `config.max_concurrent`.

        Limits how many environment processes exist simultaneously in memory. Each
        sub-batch should be fully processed (trainers created, used, torn down) before
        the next sub-batch is started.

        Parameters
        ----------
        batch : List[int]
            Full batch from `next_batch`

        Yields
        -------
        sub_batch : List[int]
            A sub-batch of up to `max_concurrent` indices
        """
        N = self.config.max_concurrent

        for i in range(0, len(batch), N):
            yield batch[i : i + N]

    def get_state(self, env_idx: int) -> "AgentSnapshot | None":
        """
        Get the serialized state for an environment.

        Parameters
        ----------
        env_idx : int
            Environment index

        Returns
        -------
        _name_ : AgentSnapshot | None
            The agent's state, or `None` if not initialized
        """
        return self._snapshots[env_idx]

    def set_state(self, env_idx: int, state: "AgentSnapshot | None") -> None:
        """
        Store a serialized agent state.

        Parameters
        ----------
        env_idx : int
            Environment index
        state : AgentSnapshot | None
            Agent state to store or `None` to clear it
        """
        self._snapshots[env_idx] = state

    def has_state(self, env_idx: int) -> bool:
        """
        Check whether an environment has a stored state.

        Parameters
        ----------
        env_idx : int
            Environment index

        Returns
        -------
        exists : bool
            `True` if a serialized state exists for this environment
        """
        return self._snapshots[env_idx] is not None

    @staticmethod
    def restore_trainer(
        trainer: "AgentTrainer",
        snapshot: "AgentSnapshot",
        device: str | None = None,
    ) -> None:
        """
        Restore a snapshot into an `AgentTrainer`.

        Parameters
        ----------
        trainer : AgentTrainer
            A trainer to restore state to
        snapshot : AgentSnapshot
            CPU-resident snapshot to restore from
        device : str (optional)
            Target device for arrays. If `None`, uses first GPU or default device.
            Default is `None`
        """
        if device is None:
            device = jax.devices()[0]

        trainer.policy_agent.update_params(
            jax.device_put(snapshot.policy_params, device)
        )
        trainer.target_agent.update_params(
            jax.device_put(snapshot.target_params, device)
        )
        trainer.value_agent.update_params(jax.device_put(snapshot.value_params, device))
        trainer.state = jax.device_put(snapshot.trainer_state, device)
