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
    start_time : float (optional)
        Start time. Default is `0.0`
    """

    hrs: float
    mins: float
    secs: float
    start_time: float = 0.0

    @classmethod
    def create(cls) -> Self:
        """
        Creates a new instance with a new start time.

        Returns
        -------
        self : Self
            A newly populated storage container
        """
        return cls(hrs=0.0, mins=0.0, secs=0.0, start_time=time.time())

    def elapsed(self) -> Self:
        """
        Calculates the elapsed time from `now` and a `start_time`.

        Returns
        -------
        self : Self
            A newly populated storage container
        """
        elapsed = time.time() - self.start_time
        hrs, remainder = divmod(elapsed, 3600)
        mins, secs = divmod(remainder, 60)

        return self.__class__(
            hrs=hrs,
            mins=mins,
            secs=secs,
            start_time=self.start_time,
        )

    def __str__(self) -> str:
        return f"{self.hrs:.2f}h {self.mins:.2f}m {self.secs:.2f}s"
