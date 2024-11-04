from abc import ABC

from pydantic import BaseModel, ConfigDict


class EnvHandler(ABC, BaseModel):
    """A base environment handler for managing RL environments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run_demo(self, episodes: int = 10) -> None:
        """
        Runs a demonstration of the environment with random actions. Strictly for initially exploring the environment.
        """
        raise NotImplementedError()
