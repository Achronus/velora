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


from dataclasses import dataclass
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from velora.nn.buffer import RolloutBatch


@dataclass
class MiniBatchData:
    """
    A batch of update-ready experience for PPO policy updates.

    Used for both the full loaded batch and the mini-batches sliced
    from it - every tensor shares the same leading sample dimension.

    Parameters
    ----------
    obs : torch.Tensor
        The observations `(n_samples, *obs_shape)`
    actions : torch.Tensor
        The agent actions `(n_samples, *act_shape)`
    log_probs : torch.Tensor
        The log probabilities of the actions `(n_samples,)`
    values : torch.Tensor
        The critic's state-value estimates `(n_samples,)`
    advantages : torch.Tensor
        The GAE advantage estimates `(n_samples,)`
    returns : torch.Tensor
        The discounted returns `(n_samples,)`
    """

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def sample(self, indices: torch.Tensor) -> "MiniBatchData":
        """
        Creates a mini-batch by slicing every field to a set of sample
        indices.

        Parameters
        ----------
        indices : torch.Tensor
            The sample indices to slice `(minibatch_size,)`

        Returns
        -------
        batch : MiniBatchData
            A mini-batch of experience `(minibatch_size, ...)`
        """
        return MiniBatchData(
            obs=self.obs[indices],
            actions=self.actions[indices],
            log_probs=self.log_probs[indices],
            values=self.values[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
        )


@dataclass
class LSTMMiniBatchData(MiniBatchData):
    """
    A batch of update-ready experience for `LSTMPPO` policy
    updates.

    Parameters
    ----------
    obs : torch.Tensor
        The observations `(n_samples, *obs_shape)`
    actions : torch.Tensor
        The agent actions `(n_samples, *act_shape)`
    log_probs : torch.Tensor
        The log probabilities of the actions `(n_samples,)`
    values : torch.Tensor
        The critic's state-value estimates `(n_samples,)`
    advantages : torch.Tensor
        The GAE advantage estimates `(n_samples,)`
    returns : torch.Tensor
        The discounted returns `(n_samples,)`
    dones : torch.Tensor
        The completion flags entering each timestep `(n_samples,)`
    initial_state : tuple[torch.Tensor, torch.Tensor]
        The `(hidden, cell)` recurrent state entering the rollout
        `(num_layers, envs_per_batch, hidden_size)`
    """

    dones: torch.Tensor
    initial_state: tuple[torch.Tensor, torch.Tensor]


class MiniBatchLoader:
    """
    An iterable over mini-batches of rollout experience.

    Load a flattened rollout with `load()`, then iterate to receive
    shuffled `MiniBatchData` slices. Each full pass over the loader
    covers a single update epoch.

    Parameters
    ----------
    batch_size : int
        The total number of samples in a loaded batch
    minibatch_size : int
        The number of samples per mini-batch
    """

    def __init__(self, batch_size: int, minibatch_size: int) -> None:
        if batch_size % minibatch_size != 0:
            raise ValueError(
                f"'batch_size' ({batch_size}) must be divisible by "
                f"'minibatch_size' ({minibatch_size})."
            )

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

        self.data: MiniBatchData | None = None

    def load(
        self,
        rollout: "RolloutBatch",
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        """
        Loads a rollout as the batch to serve mini-batches from,
        replacing any previously loaded batch.

        Parameters
        ----------
        rollout : RolloutBatch
            Flattened batch of rollout experience `(batch_size, ...)`
        advantages : torch.Tensor
            The GAE advantage estimates `(batch_size,)`
        returns : torch.Tensor
            The discounted returns `(batch_size,)`
        """
        n_samples = rollout.obs.shape[0]

        if n_samples != self.batch_size:
            raise ValueError(
                f"Incorrect sample count. Got '{n_samples}', must load '{self.batch_size}'."
            )

        if rollout.log_probs.shape != (n_samples,):
            raise ValueError(
                "Rollout must be flattened. Got 'log_probs' shape "
                f"'{tuple(rollout.log_probs.shape)}', expected '({n_samples},)'."
            )

        if advantages.shape != (n_samples,) or returns.shape != (n_samples,):
            raise ValueError(
                f"Shape mismatch. Got 'advantages' '{tuple(advantages.shape)}' and "
                f"'returns' '{tuple(returns.shape)}', expected '({n_samples},)'."
            )

        self.data = MiniBatchData(
            obs=rollout.obs,
            actions=rollout.actions,
            log_probs=rollout.log_probs,
            values=rollout.values,
            advantages=advantages,
            returns=returns,
        )

    def __iter__(self) -> Iterator[MiniBatchData]:
        """
        Iterates over the loaded batch in shuffled mini-batches,
        covering a single update epoch.

        Yields
        ------
        batch : MiniBatchData
            A mini-batch of experience `(minibatch_size, ...)`
        """
        if self.data is None:
            raise ValueError("No data added. Use 'load()' first.")

        indices = torch.randperm(self.batch_size, device=self.data.obs.device)

        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            yield self.data.sample(indices[start:end])

    def __len__(self) -> int:
        return self.batch_size if self.data is not None else 0


class LSTMMiniBatchLoader(MiniBatchLoader):
    """
    An iterable over mini-batches of rollout experience for
    `LSTMPPO`.

    Shuffles environments (not timesteps) so each mini-batch holds
    whole trajectories, keeping timesteps in order for the recurrent
    state to replay correctly.

    Parameters
    ----------
    num_steps : int
        The number of rollout timesteps per environment
    num_envs : int
        The number of parallel environments
    num_minibatches : int
        The number of mini-batches per update epoch
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_minibatches: int,
    ) -> None:
        if num_envs % num_minibatches != 0:
            raise ValueError(
                f"'num_envs' ({num_envs}) must be divisible by "
                f"'num_minibatches' ({num_minibatches}) to keep "
                "trajectories whole during mini-batching."
            )

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.envs_per_batch = num_envs // num_minibatches

        super().__init__(num_steps * num_envs, num_steps * self.envs_per_batch)

        self._dones: torch.Tensor | None = None
        self._initial_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_rollout_state(
        self,
        initial_state: tuple[torch.Tensor, torch.Tensor],
        dones: torch.Tensor,
    ) -> None:
        """
        Stores the recurrent state and completion flags entering the
        rollout, required before each `load()`.

        Parameters
        ----------
        initial_state : tuple[torch.Tensor, torch.Tensor]
            The `(hidden, cell)` recurrent state entering the rollout
            `(num_layers, num_envs, hidden_size)`
        dones : torch.Tensor
            The flattened completion flags entering each timestep
            `(batch_size,)`
        """
        if dones.shape != (self.batch_size,):
            raise ValueError(
                f"Shape mismatch. Got 'dones' '{tuple(dones.shape)}', "
                f"expected '({self.batch_size},)'."
            )

        hidden, cell = initial_state

        if hidden.shape[1] != self.num_envs or cell.shape[1] != self.num_envs:
            raise ValueError(
                "Environment count mismatch. Got 'initial_state' shapes "
                f"'{tuple(hidden.shape)}' and '{tuple(cell.shape)}', "
                f"expected '{self.num_envs}' environments at dim 1."
            )

        self._initial_state = initial_state
        self._dones = dones

    def load(
        self,
        rollout: "RolloutBatch",
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        """
        Loads a rollout as the batch to serve mini-batches from,
        replacing any previously loaded batch. Requires
        `set_rollout_state()` to have stored the rollout's recurrent
        state first.

        Parameters
        ----------
        rollout : RolloutBatch
            Flattened batch of rollout experience `(batch_size, ...)`
        advantages : torch.Tensor
            The GAE advantage estimates `(batch_size,)`
        returns : torch.Tensor
            The discounted returns `(batch_size,)`
        """
        if self._initial_state is None or self._dones is None:
            raise ValueError(
                "Missing rollout state. Use 'set_rollout_state()' before 'load()'."
            )

        super().load(rollout, advantages, returns)

    def __iter__(self) -> Iterator[LSTMMiniBatchData]:
        """
        Iterates over the loaded batch in mini-batches of shuffled
        environments, covering a single update epoch.

        Yields
        ------
        batch : LSTMMiniBatchData
            A mini-batch of whole-trajectory experience
            `(minibatch_size, ...)`, with the recurrent state sliced
            to the mini-batch's environments
        """
        if self.data is None or self._dones is None or self._initial_state is None:
            raise ValueError("No data added. Use 'load()' first.")

        device = self.data.obs.device
        env_inds = torch.randperm(self.num_envs, device=device)
        flat_inds = torch.arange(self.batch_size, device=device).reshape(
            self.num_steps,
            self.num_envs,
        )

        for start in range(0, self.num_envs, self.envs_per_batch):
            end = start + self.envs_per_batch
            mb_env_inds = env_inds[start:end]
            mb_flat_inds = flat_inds[:, mb_env_inds].reshape(-1)
            data = self.data.sample(mb_flat_inds)

            yield LSTMMiniBatchData(
                obs=data.obs,
                actions=data.actions,
                log_probs=data.log_probs,
                values=data.values,
                advantages=data.advantages,
                returns=data.returns,
                dones=self._dones[mb_flat_inds],
                initial_state=(
                    self._initial_state[0][:, mb_env_inds],
                    self._initial_state[1][:, mb_env_inds],
                ),
            )
