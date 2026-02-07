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

from typing import Dict, Self, Sequence

import jax.numpy as jnp
from flax import struct

from velora.utils.format import number_to_short


@struct.dataclass(frozen=True)
class ParamCount:
    """Parameter count for a network or agent."""

    active: int
    total: int

    def __str__(self) -> str:
        return f"{number_to_short(self.active)}/{number_to_short(self.total)}"


@struct.dataclass(frozen=True)
class RewardStatistics:
    """
    Reward statistics for completed episodes.

    Parameters
    ----------
    avg_reward : float
        Average reward
    reward_std : float
        Reward standard deviation
    reward_min : float
        Minimum reward
    reward_max : float
        Maximum reward
    """

    avg_reward: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    @classmethod
    def from_rewards(cls, rewards: Sequence[float]) -> Self:
        """
        Create statistics from a sequence of rewards.

        Parameters
        ----------
        rewards : Sequence[float]
            Sequence of reward values

        Returns
        -------
        stats : RewardStatistics
            Computed reward statistics
        """
        if not rewards:
            return cls()

        arr = jnp.array(rewards)
        return cls(
            avg_reward=float(jnp.mean(arr)),
            reward_std=float(jnp.std(arr)),
            reward_min=float(jnp.min(arr)),
            reward_max=float(jnp.max(arr)),
        )

    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary.

        Returns
        -------
        stats : Dict[str, float]
            Reward statistics as a dictionary
        """
        return vars(self)
