from abc import ABC, abstractmethod
from typing import Any, Self

from pydantic import BaseModel


class Analytics(ABC, BaseModel):
    """An analytics base class."""

    @abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> Any:
        """Starts a new run for the analytics tracker."""
        pass  # pragma: no cover

    @abstractmethod
    def log(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        """Logs metrics to the run."""
        pass  # pragma: no cover

    @abstractmethod
    def finish(self) -> None:
        """Marks the run as finished, uploads final data, and resets run instance to None."""
        pass  # pragma: no cover


class NullAnalytics(Analytics):
    """An empty analytics tracker for disabling analytics tracking."""

    def init(self, _: Any) -> Self:
        return self

    def log(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        pass

    def finish(self) -> None:
        pass
