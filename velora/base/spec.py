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

from typing import Tuple

import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass(frozen=True)
class LayerSpec:
    """
    Specification for a single Neural Circuit Policy (NCP) layer.

    Parameters
    ----------
    mask : chex.Array
        Sparse connectivity mask with polarities `(-1, 0, 1)`
    n_hidden : int
        Number of hidden units (output dimension)
    """

    mask: chex.Array
    n_hidden: int

    @property
    def shape(self) -> Tuple[int, ...]:
        return jnp.shape(self.mask)


@struct.dataclass(frozen=True)
class HeadSpec:
    """
    Base class for head specifications.
    """

    def hidden_count(self) -> int:
        """Returns the total hidden node count across all heads."""
        raise NotImplementedError()

    def hidden_sizes(self) -> Tuple[int, ...]:
        """Returns hidden sizes in head order for state splitting."""
        raise NotImplementedError()
