from abc import ABC
from typing import Any

from pydantic import BaseModel


class Analytics(ABC, BaseModel):
    """An analytics base class."""

    def init(self) -> None:
        """Initializes the project."""
        raise NotImplementedError()

    def log(self, metrics: dict[str, Any]) -> None:
        """Logs metrics."""
        raise NotImplementedError()

    def finish(self) -> None:
        """Performs analytics cleanup and terminates connection."""
        raise NotImplementedError()
