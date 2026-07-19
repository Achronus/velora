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


from typing import Tuple

import torch


class RunningMeanStd:
    """
    Tracks the running mean and variance of a batch of tensors using
    a parallel variant of Welford's algorithm.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The per-sample tensor shape
    device : torch.device
        Device to store the statistics on
    eps : float (optional)
        Initial count to avoid division by zero. Default is `1e-4`
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        eps: float = 1e-4,
    ) -> None:
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = eps

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """
        Updates the statistics with a batch of samples.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of samples `(batch_size, *shape)`
        """
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total