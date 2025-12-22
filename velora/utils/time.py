import time
from dataclasses import dataclass
from typing import Self


@dataclass
class ElapsedTime:
    """
    A storage container for time tracking.

    Parameters:
        hrs (float) hours taken
        mins (float) minutes taken
        secs (float) seconds taken
    """

    hrs: float
    mins: float
    secs: float

    @classmethod
    def elapsed(cls, start_time: float) -> Self:
        """
        Calculates the elapsed time from `now` and a `start_time`.

        Parameters:
            start_time (float): the start time of an event

        Returns:
            self (Self): a newly populated storage container.
        """
        elapsed = time.time() - start_time
        hrs, remainder = divmod(elapsed, 3600)
        mins, secs = divmod(remainder, 60)

        return cls(hrs=hrs, mins=mins, secs=secs)

    def __str__(self) -> str:
        return f"{self.hrs:.2f}h {self.mins:.2f}m {self.secs:.2f}s"
