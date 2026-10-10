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

from abc import ABC, abstractmethod

import torch


class RLSpace(ABC):
    """
    Minimal base for custom spaces, so `RLEnv` can bound its type parameters.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of a single (unbatched) element
    dtype : torch.dtype
        Element dtype
    device : torch.device | str (optional)
        Device that samples are created on. Default is `"cuda"`
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str = "cuda",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)

    @abstractmethod
    def sample(
        self,
        batch_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample from the space.

        Parameters
        ----------
        batch_size : int (optional)
            Number of samples to draw. When `None`, draws a single unbatched
            sample. Default is `None`
        generator : torch.Generator (optional)
            Random number generator to sample with. Must live on the same
            device as the space. When `None`, uses the global
            torch generator. Default is `None`

        Returns
        -------
        samples : torch.Tensor
            Sampled elements `(*shape)` when `batch_size` is `None`,
            otherwise `(batch_size, *shape)`
        """
        ...

    @abstractmethod
    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check whether elements lie within the spaces bounds (inclusive).

        Parameters
        ----------
        x : torch.Tensor
            Element(s) to check `(*shape)` or `(batch_size, *shape)`

        Returns
        -------
        mask : torch.Tensor
            Membership result, `bool`. Shape `()` for a single element,
            otherwise `(batch_size,)`. Elements containing NaN are `False`

        """
        ...
