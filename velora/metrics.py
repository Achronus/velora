from typing import Any
from pydantic import BaseModel


class EpisodeMetrics(BaseModel):
    """Contains episode metrics."""

    score: float = 0.0
    loss: float = 0.0

    def update(self, reward: float, loss: float) -> None:
        """Updates the metrics."""
        self.score += reward
        self.loss += abs(loss)

    def norm_loss(self, timesteps: int) -> None:
        """Normalizes the loss based on the number of timesteps."""
        self.loss /= timesteps


class BatchEpisodeMetrics(BaseModel):
    """Contains batch episode metrics."""

    avg_score: float = 0.0
    avg_loss: float = 0.0
    success_percent: float = 0.0
    success_count: int = 0

    def log_update(self, n_eps: int) -> None:
        """
        Updates the success percentage and avg score based on `n_eps`.

        Args:
            n_eps (int): total number of episodes
        """
        self.success_percent = (self.success_count / n_eps) * 100
        self.avg_score /= n_eps
        self.avg_loss /= n_eps

    def update(self, terminated: bool, score: float, loss: float) -> None:
        """
        Updates the metrics.

        Args:
            terminated (bool): whether the agent reached the terminal state
            score (float): a single episodes total score
        """
        if terminated:
            self.success_count += 1

        self.avg_score += score
        self.avg_loss += loss


class Metrics(BaseModel):
    """A storage container for algorithm logging metrics."""

    ep: EpisodeMetrics = EpisodeMetrics()
    batch: BatchEpisodeMetrics = BatchEpisodeMetrics()

    def norm_ep(self, timesteps: int) -> None:
        """Normalizes episode metrics."""
        return self.ep.norm_loss(timesteps)

    def ep_update(self, reward: float, loss: float) -> None:
        """Updates the episode metrics."""
        self.ep.update(reward, loss)

    def batch_update(self, terminated: bool) -> None:
        """Updates the batch metrics."""
        self.batch.update(terminated, self.ep.score, self.ep.loss)
        self.reset_ep()

    def log_update(self, n_eps: int) -> None:
        """Updates the batch metrics during logging."""
        self.batch.log_update(n_eps)

    def reset_ep(self) -> None:
        """Resets the episode metrics."""
        self.ep = EpisodeMetrics()

    def ep_dict(self) -> dict[str, Any]:
        """Returns a dictionary of the episode metrics."""
        return self.ep.model_dump()

    def __str__(self) -> str:
        return f"Batch - Avg Score: {self.batch.avg_score}, Avg Loss: {round(self.batch.avg_loss, 2)}, Terminal Reached: {self.batch.success_percent}%"
