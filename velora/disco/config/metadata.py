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
from typing import Any, Dict, List

from velora.tracking.metadata import CheckpointMetadata


@dataclass(frozen=True)
class RuleTrainerMetadata(CheckpointMetadata):
    """
    Metadata persisted alongside `RuleTrainer` checkpoints.

    Captures everything needed to reconstruct a `RuleTrainer` from a
    checkpoint directory without access to the original script.

    Parameters
    ----------
    config : Dict[str, Any]
        Serialized `RuleTrainerSettings`
    agents_per_env : int
        Number of independent agent trainers per environment
    max_group_size : int
        Maximum trainers vmapped per chunk
    num_env_workers : int
        Threads for parallel environment stepping
    seed : int
        Random number generator seed
    envs : List[Dict[str, Any]]
        Serialized environment groups
    use_bfloat16 : bool
        Whether half-precision rollout buffers are enabled
    """

    config: Dict[str, Any]
    agents_per_env: int
    max_group_size: int
    num_env_workers: int
    seed: int
    envs: List[Dict[str, Any]]
    use_bfloat16: bool
