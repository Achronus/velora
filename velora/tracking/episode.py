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
from typing import Any, Dict, Tuple


@dataclass
class EpisodeTracker:
    """
    Tracks completed episode statistics during trajectory collection.

    Accumulates returns and lengths from environment info dicts and
    exposes summary metrics for logging.

    Parameters
    ----------
    completed_returns : Tuple[float, ...]
        Episodic returns for all completed episodes. Default is `()`
    completed_lengths : Tuple[int, ...]
        Episode lengths for all completed episodes. Default is `()`
    """

    completed_returns: Tuple[float, ...] = ()
    completed_lengths: Tuple[int, ...] = ()

    def record(self, info: Dict[str, Any]) -> None:
        """
        Record completed episode statistics from an env info dict.

        Parameters
        ----------
        info : Dict[str, Any]
            Environment metadata returned by `envs.step()`.
            Expects a `"final_info"` key populated by
            `RecordEpisodeStatistics`.
        """
        if "final_info" not in info:
            return

        for env_info in info["final_info"]:
            if env_info is not None and "episode" in env_info:
                self.completed_returns += (env_info["episode"]["r"],)
                self.completed_lengths += (env_info["episode"]["l"],)

    def reset(self) -> None:
        """Clear accumulated episode data for the next collection call."""
        self.completed_returns = ()
        self.completed_lengths = ()

    def metrics(self) -> Dict[str, float] | None:
        """
        Return episode metrics if any episodes completed this collection.

        Returns
        -------
        metrics : Dict[str, float] | None
            Summary metrics dict, or `None` if no episodes completed.
        """
        if not self.completed_returns:
            return None

        return {
            "episode/reward_mean": self.mean_return,
            "episode/reward_min": float(min(self.completed_returns)),
            "episode/reward_max": float(max(self.completed_returns)),
            "episode/length_mean": self.mean_length,
            "episode/count": self.num_completed,
        }

    @property
    def num_completed(self) -> int:
        """Number of episodes completed in the current collection."""
        return len(self.completed_returns)

    @property
    def mean_return(self) -> float:
        """Mean episodic return across completed episodes."""
        if not self.completed_returns:
            return 0.0
        return sum(self.completed_returns) / len(self.completed_returns)

    @property
    def mean_length(self) -> float:
        """Mean episode length across completed episodes."""
        if not self.completed_lengths:
            return 0.0
        return sum(self.completed_lengths) / len(self.completed_lengths)
