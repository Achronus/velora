from datetime import datetime
from typing import List

from sqlmodel import Field, Relationship, SQLModel


class Experiment(SQLModel, table=True):
    """
    Experiment information tracking agent, environment, and metadata.

    Attributes:
        id (int): unique identifier for the experiment
        agent (str): the name of the agent used in the experiment
        env (str): the name of the environment used in the experiment
        config (str): a JSON string containing the agent's configuration details
        created_at (datetime): the date and time the experiment was created
    """

    id: int | None = Field(default=None, primary_key=True)
    agent: str = Field(index=True)
    env: str = Field(index=True)
    config: str  # JSON object
    created_at: datetime = Field(default_factory=datetime.now)

    # Relationships
    episodes: List["Episode"] = Relationship(back_populates="experiment")


class Episode(SQLModel, table=True):
    """
    Episode-level metrics tracking reward, length, and agent performance.

    Attributes:
        id (int): a unique identifier for the episode
        experiment_id (int): the experiment ID associated to the episode
        episode_num (int): the episode index
        reward (float): the episodic reward (return)
        length (int): the number of timesteps performed to terminate the episode
        reward_moving_avg (float): the episodes reward moving average based on a window size
        reward_moving_std (float): the episodes reward moving standard deviation based on a
            window size
        actor_loss (float): the average Actor loss for the episode
        critic_loss (float): the average Critic loss for the episode
        entropy_loss (float): the average Entropy loss for the episode
        created_at (datetime): the date and time when the the entry was created
    """

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
    entropy_loss: float = Field(default=0.0)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

    # Relationships
    experiment: Experiment = Relationship(back_populates="episodes")
