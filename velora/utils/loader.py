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


from collections.abc import Iterator
from dataclasses import dataclass
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
