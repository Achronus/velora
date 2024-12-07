from pydantic import BaseModel


class Metrics(BaseModel):
    """A storage container for algorithm logging metrics."""

    score: float = 0.0
    avg_score: float = 0.0
    loss: float = 0.0
    ep_success_percent: int = 0
    ep_success_count: int = 0

    def update_scores(self, reward: float, loss: float) -> None:
        """Updates the `score` and `loss` values."""
        self.score += reward
        self.loss += loss

    def update_percent(self, n_eps: int) -> None:
        """
        Updates the success percentage.

        Args:
            n_eps (int): total number of episodes
        """
        self.ep_success_percent = (self.ep_success_count / n_eps) * 100

    def update_ep_counts(self, terminated: bool) -> None:
        """
        Updates episode specific metrics.

        Args:
            terminated (bool): whether the agent reached the terminal state
        """
        if terminated:
            self.ep_success_count += 1

        self.avg_score += self.score

    def reset_ep_counts(self) -> None:
        """Resets episode specific metrics back to 0."""
        self.avg_score = 0.0
        self.ep_success_count = 0

    def __str__(self) -> str:
        return f"Score: {self.score} | Loss: {self.loss} | Episode Iterations - Avg Score: {self.avg_score}, Success Rate: {self.ep_success_percent}%"
