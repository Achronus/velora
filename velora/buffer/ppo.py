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

import torch

from velora.buffer.base import BufferSamples, RLBuffer


@dataclass(frozen=True)
class PPOSamples(BufferSamples):
    """
    A batch of samples from a PPO buffer.

    Tensors are shaped `(num_steps, num_envs, *shape)` for a full
    rollout, or `(batch_size, *shape)` for flattened mini-batches
    from `PPOBuffer.sample()`.

    Parameters
    ----------
    obs : torch.Tensor
        The environment observations `(..., *obs_shape)`
    actions : torch.Tensor
        The agent actions `(..., *act_shape)`
    rewards : torch.Tensor
        The rewards received from the environments `(..., 1)`
    dones : torch.Tensor
        The environment completion flags, `1` if the episode ended
        after the step `(..., 1)`
    log_probs : torch.Tensor
        The log probabilities of the actions under the policy
        that collected them `(..., 1)`
    values : torch.Tensor
        The critic's state-value estimates `(..., 1)`
    advantages : torch.Tensor
        The GAE(λ) advantage estimates `(..., 1)`
    returns : torch.Tensor
        The λ-returns, `advantages + values`, used as critic targets
        `(..., 1)`
    """

    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class PPOBuffer(RLBuffer):
    """
    A PPO rollout buffer that stores one rollout, computes GAE(λ)
    advantages over it, and yields shuffled mini-batches.

    Usage per rollout: `add()` each step, then `compute_gae()`,
    then `sample()` once per epoch, and lastly `reset()` before
    the next rollout.

    Parameters
    ----------
    capacity : int
        Number of timesteps per environment in one rollout
    obs_shape : tuple[int, ...]
        The shape of a single environment observation space
    act_shape : tuple[int, ...]
        The shape of a single environment action space
    gamma : float
        The discount factor
    gae_lambda : float
        The lambda for the GAE
    device : torch.device
        Device to load tensors onto
    num_envs : int (optional)
        Number of parallel environments used. Default is `1`
    """

    def __init__(
        self,
        capacity: int,
        *,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...],
        gamma: float,
        gae_lambda: float,
        device: torch.device,
        num_envs: int = 1,
    ) -> None:
        super().__init__(capacity, device=device, num_envs=num_envs)

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._is_gae_computed = False

        # Storage
        self.obs = torch.zeros(
            (capacity, num_envs) + obs_shape,
            device=device,
        )
        self.actions = torch.zeros(
            (capacity, num_envs) + act_shape,
            device=device,
        )
        self.rewards = torch.zeros((capacity, num_envs, 1), device=device)
        self.dones = torch.zeros((capacity, num_envs, 1), device=device)

        self.log_probs = torch.zeros((capacity, num_envs, 1), device=device)
        self.values = torch.zeros((capacity, num_envs, 1), device=device)
        self.advantages = torch.zeros((capacity, num_envs, 1), device=device)
        self.returns = torch.zeros((capacity, num_envs, 1), device=device)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Add a set of experience to the buffer.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of environment observations `(num_envs, *obs_shape)`
        actions : torch.Tensor
            A batch of agent actions `(num_envs, *act_shape)`
        rewards : torch.Tensor
            The rewards received from the environments `(num_envs, 1)`
        dones : torch.Tensor
            The environment completion flags, `1` if the episode
            ended after this step (terminated or truncated)
            `(num_envs, 1)`
        log_probs : torch.Tensor
            The log probabilities of the actions `(num_envs, 1)`
        values : torch.Tensor
            The critic's state-value estimates `(num_envs, 1)`

        Raises
        ------
        buffer_full : ValueError
            Error when buffer has reached capacity
        """
        if self.full:
            raise ValueError("Buffer is already full.")

        self.obs[self._position] = obs
        self.actions[self._position] = actions
        self.rewards[self._position] = rewards
        self.dones[self._position] = dones

        self.log_probs[self._position] = log_probs
        self.values[self._position] = values

        self._advance()
        self._is_gae_computed = False

    def sample(self, mini_batches: int) -> Iterator[PPOSamples]:
        """
        Yield mini-batch samples from the buffer.

        Parameters
        ----------
        mini_batches : int
            Number of mini-batches to split the rollout into

        Returns
        -------
        samples : Iterator[PPOSamples]
            A generator yielding `mini_batches` of shuffled batches,
            each of shape `(batch_size, *shape)`. Call once per epoch
            for a fresh shuffle

        Raises
        ------
        gae_not_computed : RuntimeError
            Error when `compute_gae()` hasn't been called for the
            current rollout
        invalid_mini_batches : ValueError
            Error when `mini_batches` is outside `[1, len(self) * num_envs]`
        """
        if not self._is_gae_computed:
            raise RuntimeError("Call 'compute_gae()' before sampling.")

        n_samples = len(self) * self.num_envs

        if not 0 < mini_batches <= n_samples:
            raise ValueError(
                f"'mini_batches' must be in the range [1, {n_samples}]. Got: {mini_batches}."
            )

        return self._iter_batches(mini_batches)

    def _iter_batches(self, mini_batches: int) -> Iterator[PPOSamples]:
        """
        Utility method for creating a generator over mini-batches
        of buffer samples.

        Enables validation conditions to be used before running.

        Parameters
        ----------
        mini_batches : int
            Number of mini-batches to split the rollout into

        Yields
        -------
        samples : PPOSamples
            A shuffled mini-batch of `(batch_size, *shape)` samples
        """
        size = len(self)

        samples = PPOSamples(
            obs=self.obs[:size],
            actions=self.actions[:size],
            rewards=self.rewards[:size],
            dones=self.dones[:size],
            log_probs=self.log_probs[:size],
            values=self.values[:size],
            advantages=self.advantages[:size],
            returns=self.returns[:size],
        ).flatten()

        # Shuffle
        indices = torch.randperm(size * self.num_envs, device=self.device)

        for idx in indices.tensor_split(mini_batches):
            yield samples.select(idx)

    def reset(self) -> None:
        super().reset()
        self._is_gae_computed = False

    @torch.no_grad()
    def compute_gae(self, last_values: torch.Tensor) -> None:
        """
        Computes the advantages and returns for a rollout using
        Generalized Advantage Estimation (GAE), bootstrapping the final
        step from `last_values`.

        Parameters
        ----------
        last_values : torch.Tensor
            The critic's state-value estimates for the observation
            following the rollout's final step `(num_envs, 1)`

        Raises
        ------
        shape_mismatch : ValueError
            Error when `last_values` doesn't have shape `(num_envs, 1)`
        empty_buffer : RuntimeError
            Error when the buffer contains no samples
        """
        size = len(self)

        if last_values.shape != self.values.shape[1:]:
            raise ValueError(
                f"'last_values' must have shape {tuple(self.values.shape[1:])}. Got: {tuple(last_values.shape)}."
            )

        if size == 0:
            raise RuntimeError("Cannot compute GAE on an empty buffer.")

        last_gae_lambda = torch.zeros_like(last_values)

        for t in reversed(range(size)):
            next_non_terminal = 1.0 - self.dones[t]
            next_values = last_values if t == size - 1 else self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae_lambda = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            )
            self.advantages[t] = last_gae_lambda

        # Store values in buffer for sampling
        self.returns[:size] = self.advantages[:size] + self.values[:size]
        self._is_gae_computed = True
