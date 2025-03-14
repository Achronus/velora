from datetime import datetime
from typing import List

from sqlmodel import Field, Relationship, SQLModel


class Experiment(SQLModel, table=True):
    """Experiment information tracking agent, environment, and metadata."""

    id: int | None = Field(default=None, primary_key=True)
    agent: str = Field(index=True)
    env: str = Field(index=True)
    config: str  # JSON object
    created_at: datetime = Field(default_factory=datetime.now)

    # Relationships
    episodes: List["Episode"] = Relationship(back_populates="experiment")
    steps: List["Step"] = Relationship(back_populates="experiment")


class Episode(SQLModel, table=True):
    """Episode-level metrics tracking reward, length, and agent performance."""

    id: int | None = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiment.id", index=True)
    episode_num: int = Field(index=True)

    # Core metrics
    reward: float
    length: int

    # Statistical metrics
    reward_moving_avg: float
    reward_moving_std: float

    # Loss metrics
    actor_loss: float
    critic_loss: float

    # Behaviour metrics
    # explore_rate: float
    # exploit_mean: float
    # exploit_std: float

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

    # Relationships
    experiment: Experiment = Relationship(back_populates="episodes")
    steps: List["Step"] = Relationship(back_populates="episode")


class Step(SQLModel, table=True):
    """Step-level metrics tracking individual actions and states."""

    id: int | None = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiment.id", index=True)
    episode_id: int = Field(foreign_key="episode.id", index=True)
    step_num: int

    # Action metrics
    action: str  # JSON object

    # Loss metrics
    actor_loss: float
    critic_loss: float

    # Behaviour metrics
    # explore_rate: float
    # exploit_mean: float
    # exploit_std: float
    # is_exploration: bool

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

    # Relationships
    experiment: Experiment = Relationship(back_populates="steps")
    episode: Episode = Relationship(back_populates="steps")
