from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from velora.config import EnvironmentSettings


class EnvHandler(ABC, BaseModel):
    """A base environment handler for managing RL environments."""

    config: EnvironmentSettings

    _env: Any = PrivateAttr(...)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def env(self) -> Any:
        """Returns the environment."""
        return self._env

    @abstractmethod
    def run_demo(self, episodes: int = 10) -> None:
        """
        Runs a demonstration of the environment with random actions. Strictly for initially exploring the environment.
        """
        pass  # pragma: no cover
