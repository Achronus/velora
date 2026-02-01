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

from flax import struct

from velora.base.spec import HeadSpec, LayerSpec


@struct.dataclass(frozen=True)
class ACMHeadSpec(HeadSpec):
    """
    Output head specification for the Action-Conditional Model (ACM).

    Parameters
    ----------
    z : LayerSpec
        Action-conditioned prediction head
    aux_pi : LayerSpec
        Auxiliary policy prediction head
    q : LayerSpec
        Action-value prediction head
    """

    z: LayerSpec
    aux_pi: LayerSpec
    q: LayerSpec

    def hidden_count(self) -> int:
        return self.z.n_hidden + self.aux_pi.n_hidden + self.q.n_hidden

    def hidden_sizes(self) -> Tuple[int, ...]:
        return (self.z.n_hidden, self.aux_pi.n_hidden, self.q.n_hidden)


@struct.dataclass(frozen=True)
class OCMHeadSpec(HeadSpec):
    """
    Output head specification for the Observation-Conditional Model (OCM).

    Parameters
    ----------
    pi : LayerSpec
        Policy prediction head
    y : LayerSpec
        Observation-conditioned prediction head
    """

    pi: LayerSpec
    y: LayerSpec

    def hidden_count(self) -> int:
        return self.pi.n_hidden + self.y.n_hidden

    def hidden_sizes(self) -> Tuple[int, ...]:
        return (self.pi.n_hidden, self.y.n_hidden)


@struct.dataclass(frozen=True)
class DiscoHeadSpec(HeadSpec):
    """
    Output head specification for the Disco Meta-Network.

    Parameters
    ----------
    pi : LayerSpec
        Policy targets head
    y : LayerSpec
        Observation-conditioned targets head
    z : LayerSpec
        Action-conditioned targets head
    """

    pi: LayerSpec
    y: LayerSpec
    z: LayerSpec

    def hidden_count(self) -> int:
        return self.pi.n_hidden + self.y.n_hidden + self.z.n_hidden

    def hidden_sizes(self) -> Tuple[int, ...]:
        return (self.pi.n_hidden, self.y.n_hidden, self.z.n_hidden)
