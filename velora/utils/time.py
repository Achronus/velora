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

import time
from dataclasses import dataclass
from typing import Self


@dataclass
class ElapsedTime:
    """
    A storage container for time tracking.

    Parameters
    ----------
    hrs : float
        Hours taken
    mins : float
        Minutes taken
    secs : float
        Seconds taken
    """

    hrs: float
    mins: float
    secs: float

    @classmethod
    def elapsed(cls, start_time: float) -> Self:
        """
        Calculates the elapsed time from `now` and a `start_time`.

        Parameters
        ----------
        start_time : float
            The start time of an event

        Returns
        -------
        self : Self
            A newly populated storage container
        """
        elapsed = time.time() - start_time
        hrs, remainder = divmod(elapsed, 3600)
        mins, secs = divmod(remainder, 60)

        return cls(hrs=hrs, mins=mins, secs=secs)

    def __str__(self) -> str:
        return f"{self.hrs:.2f}h {self.mins:.2f}m {self.secs:.2f}s"
