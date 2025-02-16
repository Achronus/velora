from copy import deepcopy

import torch


class OUNoise:
    """
    Implements the Ornstein-Uhlenbeck process for generating noise.

    Often used in Reinforcement Learning to encourage exploration.

    Parameters:
        size (int): size of the sample tensor (e.g., action size).
        mu (float, optional): The mean value to which the noise gravitates.
            Defaults to `0.0`.
        theta (float, optional): The speed of mean reversion. Defaults to `0.15`.
        sigma (float, optional): The scale of the random component.
            Defaults to `0.2`.
        device (torch.device, optional): the device for computation.
            Defaults to `cpu`
    """

    def __init__(
        self,
        size: int,
        *,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        device: torch.device = "cpu",
    ) -> None:
        self.mu = torch.full((size,), mu, dtype=torch.float32, device=device)
        self.theta = theta
        self.sigma = sigma
        self.device = device

        self.state = None

        self.reset()

    def reset(self) -> None:
        """Resets the noise process to the mean value."""
        self.state = deepcopy(self.mu)

    def sample(self) -> torch.Tensor:
        """
        Generates a new noise sample using the Ornstein-Uhlenbeck process.

        Returns:
            torch.Tensor: A noise sample of the same shape as `mu`.
        """
        t = torch.randn(self.state.shape, dtype=torch.float32, device=self.device)
        dx = self.theta * (self.mu - self.state) + self.sigma * t

        self.state += dx
        return self.state
