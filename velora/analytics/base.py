from abc import ABC
from typing import Any

from pydantic import BaseModel


class Analytics(ABC, BaseModel):
    """An analytics base class."""

    def init(self) -> None:
        """Starts a new run for the analytics tracker."""
        raise NotImplementedError()

    def log(self, metrics: dict[str, Any]) -> None:
        """Logs metrics to the run."""
        raise NotImplementedError()

    def finish(self) -> None:
        """Marks the run as finished, uploads final data, and resets run instance to None."""
        raise NotImplementedError()
