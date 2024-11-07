from pathlib import Path

from pydantic import BaseModel

from velora.agent import Agent
from velora.env import EnvHandler


class Controller(BaseModel):
    """

    Args:
        config_filepath (pathlib.Path | str):
        handler (velora.env.EnvHandler):
        agent (velora.agent.Agent):
    """

    config_filepath: Path | str
    handler: EnvHandler
    agent: Agent

    def train(self) -> None:
        """Trains the agent."""
        pass

    def predict(self) -> None:
        """Uses the trained agent to make a prediction."""
        pass

    def plot(self) -> None:
        """Plots a metric."""
        pass
