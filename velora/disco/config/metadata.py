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
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class RuleTrainerMetadata:
    """
    Metadata persisted alongside `RuleTrainer` runs.

    Parameters
    ----------
    config : Dict[str, Any]
        Serialized `RuleTrainerSettings`
    num_envs : int
        Number of unique environments trained on
    num_trainers : int
        Number of trainers used during training
    agents_per_env : int
        Number of independent agent trainers per environment
    n_meta_steps : int
        Number of meta-steps run during training
    seed : int
        Random number generator seed
    env_names : List[str]
        Unique canonical envrax env names (e.g. `"mjx/hopper_hop-v0"`)
    env_categories : Dict[str, int]
        Mapping of environment suite category → environment count (e.g.
        `{"MuJoCo Playground": 25}`)
    """

    config: Dict[str, Any]
    num_envs: int
    num_trainers: int
    agents_per_env: int
    n_meta_steps: int
    seed: int
    env_names: List[str]
    env_categories: Dict[str, int]


@dataclass(frozen=True)
class SlotMetadata:
    """
    Per trainer host-side static metadata.

    Parameters
    ----------
    env_key : str
        Multi-env dictionary key
    env_name : str
        Human-readable environment name
    obs_dim : int
        Environment observation dimension (unpadded)
    action_dim : int
        Environment action dimension (unpadded)
    """

    env_key: str
    env_name: str
    obs_dim: int
    action_dim: int


@dataclass(frozen=True)
class PoolMetadata:
    """
    Pool trainer static metadata.

    Parameters
    ----------
    slots : Tuple[SlotMetadata]
        Per trainer metadata
    max_obs_dim : int
        Maximum observation size
    max_action_dim : int
        Maximum action space size
    """

    slots: Tuple[SlotMetadata, ...]
    max_obs_dim: int
    max_action_dim: int

    @property
    def num_trainers(self) -> int:
        return len(self.slots)
