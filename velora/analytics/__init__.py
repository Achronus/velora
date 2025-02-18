from pydantic import BaseModel, Field

from velora.analytics.base import Analytics, NullAnalytics
from velora.analytics.wandb import WeightsAndBiases

__all__ = [
    "Analytics",
    "WeightsAndBiases",
    "NullAnalytics",
]


class EpisodeStats(BaseModel):
    """
    Stores episode statistics as a model.

    Args:
        reward (float): cumulative reward
        size (int): episode length
        time_elapsed (float): elapsed time since beginning of episode
    """

    reward: float = Field(..., alias="r")
    size: int = Field(..., alias="l")
    time_elapsed: float = Field(..., alias="t")
