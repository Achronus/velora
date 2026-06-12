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

from dataclasses import dataclass
from typing import Tuple

from flax import nnx, struct

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


@dataclass(frozen=True)
class EncoderSpec:
    """
    Static metadata for observation encoders.

    Parameters
    ----------
    graphdef : nnx.GraphDef
        The encoders structural definition
    rest : nnx.State
        Static non-`Param` state
    encoding_dim : int
        Encoding output dimension
    max_obs_dim : int
        Padded observation width size
    """

    graphdef: nnx.GraphDef
    rest: nnx.State

    encoding_dim: int
    max_obs_dim: int


@dataclass(frozen=True)
class PolicySpec:
    """
    Static metadata for policy agents.

    Parameters
    ----------
    graphdef : nnx.GraphDef
        The agents structural definition
    rest : nnx.State
        Static non-`Param` state
    ocm_hidden_size : int
        OCM hidden dimension
    acm_hidden_size : int
        ACM hidden dimension
    max_action_size : int
        Maximum action size
    """

    graphdef: nnx.GraphDef
    rest: nnx.State

    ocm_hidden_size: int
    acm_hidden_size: int
    max_action_dim: int


@dataclass(frozen=True)
class DiscoSpec:
    """
    Static metadata for disco agents.

    Parameters
    ----------
    graphdef : nnx.GraphDef
        The agents structural definition
    rest : nnx.State
        Static non-`Param` state
    disco_hidden_size : int
        Disco network hidden size
    meta_hidden_size : int
        Meta LNN hidden size
    is_frozen : bool
        Frozen parameter flag
    """

    graphdef: nnx.GraphDef
    rest: nnx.State

    disco_hidden_size: int
    meta_hidden_size: int
    is_frozen: bool


@dataclass(frozen=True)
class DiscoValueSpec:
    """
    Static metadata for disco value agents.

    Parameters
    ----------
    graphdef : nnx.GraphDef
        The agents structural definition
    rest : nnx.State
        Static non-`Param` state
    hidden_size : int
        Hidden state dimension
    """

    graphdef: nnx.GraphDef
    rest: nnx.State

    hidden_size: int


@dataclass(frozen=True)
class Specs:
    """
    Storage for each rule trainer specification containing static metadata.

    Parameters
    ----------
    encoder : EncoderSpec
        Encoder specification
    policy : PolicySpec
        Policy agent specification
    disco : DiscoSpec
        Disco agent specification
    value : DiscoValueSpec
        Value disco agent specification
    """

    encoder: EncoderSpec
    policy: PolicySpec
    disco: DiscoSpec
    value: DiscoValueSpec
