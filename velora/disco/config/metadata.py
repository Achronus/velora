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

from dataclasses import dataclass, field
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
    seed : int
        Random number generator seed
    env_names : List[str]
        Canonical envrax env names (e.g. `["mjx/hopper_hop-v0", ...]`).
        Stored unique (before `agents_per_env` multiplication).
    env_categories : Dict[str, int]
        Mapping of suite category → environment count (e.g.
        `{"MuJoCo Playground": 25}`). Used by `RuleTrainer.restore` to
        reconstruct the dashboard's per-suite breakdown without
        re-querying the envrax registry.
    disco_key : List[int]
        Serialized JAX RNG key for the DiscoAgent. Used by
        `DiscoAgent.load()` to reconstruct the agent from a run directory
    parent_run : str | None (optional)
        Path to the parent run directory this training was extended from.
        `None` for fresh runs. Default is `None`
    parent_checkpoint_step : int | None (optional)
        Checkpoint step restored from in the parent run.
        `None` for fresh runs. Default is `None`
    """

    config: Dict[str, Any]
    agents_per_env: int
    seed: int
    env_names: List[str]
    env_categories: Dict[str, int]
    disco_key: List[int]
    parent_run: str | None = field(default=None)
    parent_checkpoint_step: int | None = field(default=None)
