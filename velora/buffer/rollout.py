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

import torch

from velora.buffer.base import BufferBase, ExperienceBatch


@dataclass(frozen=True)
class RolloutBatch(ExperienceBatch):
    """
    A batch of experience for a full rollout.

    Parameters
    ----------
    obs : torch.Tensor
        The environment observations `(capacity, n_envs, *obs_shape)`
    actions : torch.Tensor
        The agent actions `(capacity, n_envs, *act_shape)`
    rewards : torch.Tensor
        The rewards received from the environments `(capacity, n_envs)`
    dones : torch.Tensor
        The environment completion flags `(capacity, n_envs)`
    log_probs : torch.Tensor
        The log probabilities of the actions `(capacity, n_envs)`
    values : torch.Tensor
        The critic's state-value estimates `(capacity, n_envs)`
    """

    log_probs: torch.Tensor
    values: torch.Tensor


class RolloutBuffer(BufferBase):
    """
    A buffer to store rollouts of experience.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments
    capacity : int
        Maximum size of the rollout buffer
    obs_shape : tuple[int, ...]
        A single environments observation space shape
    act_shape : tuple[int, ...]
        A single environments action space shape
    device : torch.device
        Device to load tensors onto
    """

    def __init__(
        self,
        n_envs: int,
        capacity: int,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...],
        *,
        device: torch.device,
    ) -> None:
        super().__init__(capacity, device=device)

        self.obs = torch.zeros((capacity, n_envs) + obs_shape).to(device)
        self.actions = torch.zeros((capacity, n_envs) + act_shape).to(device)
        self.log_probs = torch.zeros((capacity, n_envs)).to(device)
        self.rewards = torch.zeros((capacity, n_envs)).to(device)
        self.dones = torch.zeros((capacity, n_envs)).to(device)
        self.values = torch.zeros((capacity, n_envs)).to(device)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Add a set of experience to the buffer.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of environment observations `(n_envs, *obs_shape)`
        actions : torch.Tensor
            A batch of agent actions `(n_envs, *act_shape)`
        log_probs : torch.Tensor
            The log probabilities of the actions `(n_envs,)`
        rewards : torch.Tensor
            The rewards received from the environments `(n_envs,)`
        dones : torch.Tensor
            The environment completion flags `(n_envs,)`
        values : torch.Tensor
            The critic's state-value estimates `(n_envs,)`

        Raises
        ------
        buffer_full : ValueError
            Error when buffer has reached capacity
        """
        if self.full:
            raise ValueError("Buffer is already full.")

        self.obs[self._position] = obs
        self.actions[self._position] = actions
        self.log_probs[self._position] = log_probs
        self.rewards[self._position] = rewards
        self.dones[self._position] = dones
        self.values[self._position] = values

        self._position += 1
        self._size += 1

    def sample(self) -> RolloutBatch:
        """
        Retrieve all samples from the buffer as a batch of experience.

        Returns
        -------
        rollouts : RolloutBatch
            Full batch of rollout experience

        Raises
        ------
        buffer_error : ValueError
            Buffer must be filled with `add()` first
        """
        if self._position < self.capacity:
            raise ValueError("Buffer must be full first.")

        return RolloutBatch(
            obs=self.obs,
            actions=self.actions,
            log_probs=self.log_probs,
            rewards=self.rewards,
            dones=self.dones,
            values=self.values,
        )

    def reset(self) -> None:
        """Resets the buffer back to its initial state."""
        self.obs.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()

        self._position = 0
