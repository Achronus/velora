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

from collections import deque
from typing import Dict, Tuple

import numpy as np


class EpisodeTracker:
    """
    Tracks completed episode statistics across a vectorized environment.

    Maintains per-environment running accumulators for the current episode
    and records completed episodes on termination.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments
    window : int (optional)
        Lifetime sliding window size. Default is `100` meta-steps
    """

    def __init__(self, num_envs: int, window: int = 100) -> None:
        self.num_envs = num_envs

        # Running accumulators - (num_envs,)
        self._current_returns = np.zeros(num_envs, dtype=np.float32)
        self._current_lengths = np.zeros(num_envs, dtype=np.int32)

        # Completed episode statistics for this collection window
        self.completed_returns: Tuple[float, ...] = ()
        self.completed_lengths: Tuple[int, ...] = ()

        # Lifetime sliding window — never reset, survives across meta-steps
        self._return_window: deque[float] = deque(maxlen=window)
        self._length_window: deque[int] = deque(maxlen=window)

    def record(
        self,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        """
        Update accumulators for one environment step.

        Parameters
        ----------
        rewards : np.ndarray
            Raw environment rewards `(num_envs,)`
        terminated : np.ndarray
            Terminal flags `(num_envs,)`
        truncated : np.ndarray
            Truncated flags `(num_envs,)`
        """
        self._current_returns += rewards.squeeze()
        self._current_lengths += 1

        done = terminated | truncated

        if done.any():
            for i in np.where(done)[0]:
                r = float(self._current_returns[i])
                l = int(self._current_lengths[i])

                self.completed_returns += (r,)
                self.completed_lengths += (l,)
                self._return_window.append(r)
                self._length_window.append(l)

                self._current_returns[i] = 0.0
                self._current_lengths[i] = 0

    def reset(self) -> None:
        """Clear accumulated episode data for the next collection window."""
        self.completed_returns = ()
        self.completed_lengths = ()

    def metrics(self) -> Dict[str, float] | None:
        """
        Return summary metrics if any episodes completed this window.

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

    @property
    def windowed_mean_return(self) -> float:
        """Sliding window mean return."""
        if not self._return_window:
            return 0.0

        return float(sum(self._return_window) / len(self._return_window))

    @property
    def windowed_mean_length(self) -> float:
        """Sliding window mean length."""
        if not self._length_window:
            return 0.0

        return float(sum(self._length_window) / len(self._length_window))
