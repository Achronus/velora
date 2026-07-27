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

import torch

from velora.sampler.base import DeterministicSampler


class WhiteNoiseSampler(DeterministicSampler):
    """
    An action sampler applying white (uncorrelated) Gaussian noise to
    deterministic policy outputs, based on
    [TD3](https://arxiv.org/abs/1802.09477).

    For TD3's configuration:
    - Exploration - (`noise_std=0.1`, unclipped)
    - Target policy smoothing - (`noise_std=0.2`, `noise_clip=0.5`)

    All returned actions are clamped to `[-1, 1]`.

    Parameters
    ----------
    noise_std : float (optional)
        The Gaussian noise standard deviation. When `0.0`,
        samples are deterministic. Default is `0.0`
    noise_clip : float (optional)
        The noise clipping bound, applied before the action bound
        clamp. When `None`, noise is unclipped. Default is `None`
    """

    def __init__(self, noise_std: float = 0.0, noise_clip: float | None = None) -> None:
        self.noise_std = noise_std
        self.noise_clip = noise_clip

    def deterministic(self, mean: torch.Tensor) -> torch.Tensor:
        """
        Selects the policy's deterministic actions.

        Parameters
        ----------
        mean : torch.Tensor
            The deterministic policy outputs
            `(batch_size, out_features)`

        Returns
        -------
        actions : torch.Tensor
            The deterministic actions, clamped to `[-1, 1]`
            `(batch_size, out_features)`
        """
        return mean.clamp(-1.0, 1.0)

    def sample(self, mean: torch.Tensor) -> torch.Tensor:
        """
        Selects exploratory actions by applying white Gaussian noise
        to the policy's deterministic outputs.

        Parameters
        ----------
        mean : torch.Tensor
            The deterministic policy outputs
            `(batch_size, out_features)`

        Returns
        -------
        actions : torch.Tensor
            The noisy actions, clamped to `[-1, 1]`
            `(batch_size, out_features)`
        """
        if self.noise_std > 0.0:
            noise = torch.randn_like(mean) * self.noise_std

            if self.noise_clip is not None:
                noise = noise.clamp(-self.noise_clip, self.noise_clip)

            mean = mean + noise

        return mean.clamp(-1.0, 1.0)
