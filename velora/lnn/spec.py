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

import jax.numpy as jnp
from flax import struct

from velora.base.spec import HeadSpec, LayerSpec


@struct.dataclass(frozen=True)
class NCPWiringSpec:
    """
    NCP wiring specification.

    Parameters
    ----------
    inter : LayerSpec
        Inter (input) layer specification
    command : LayerSpec
        Command (hidden) layer specification
    motor : HeadSpec
        Motor (output) head specification
    """

    inter: LayerSpec
    command: LayerSpec
    motor: HeadSpec

    @property
    def hidden_size(self) -> int:
        """
        Total hidden state size across all layers.

        Returns
        -------
        n_hidden : int
            Number of hidden nodes in the NCP
        """
        return self.inter.n_hidden + self.command.n_hidden + self.motor.hidden_count()

    def h_split_indices(self) -> Tuple[int, ...]:
        """
        Compute cumulative indices for splitting the hidden state.

        Returns
        -------
        hidden_indices : Tuple[int, ...]
            Hidden state split indices, one value per layer
        """
        sizes = [
            self.inter.n_hidden,
            self.command.n_hidden,
            *self.motor.hidden_sizes(),
        ]
        return tuple(jnp.cumsum(jnp.array(sizes[:-1])))


@struct.dataclass(frozen=True)
class SingleHeadSpec(HeadSpec):
    """
    Output head specification for a standard motor layer with one output head.

    Parameters
    ----------
    out : LayerSpec
        Motor prediction head
    """

    out: LayerSpec

    def hidden_count(self) -> int:
        return self.out.n_hidden

    def hidden_sizes(self) -> Tuple[int, ...]:
        return (self.out.n_hidden,)
