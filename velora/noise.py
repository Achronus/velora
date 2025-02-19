from copy import deepcopy

import torch


class OUNoise:
    """
    Implements the [Ornstein-Uhlenbeck](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823) (OU) process for generating noise. Often used in Reinforcement Learning to encourage exploration.

    Uses the approach discussed in the DDPG paper: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971).

    $$
    X_{t+1} = X_{t} + \\theta (\\mu - X_{t}) + \\sigma \\xi_{t}
    $$

    Where:

    - $X_t$ is the current noise state.
    - $\\theta$ is the mean-reverting factor (controls how quickly the noise returns to $\\mu$).
    - $\\mu$ is the long-term mean.
    - $\\sigma$ is the noise scale.
    - $\\xi_t \\sim \\mathcal{N}(0,1)$ is standard Gaussian noise.
    """

    def __init__(
        self,
        size: int,
        *,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Parameters:
            size (int): size of the sample tensor (e.g., action size)
            mu (float, optional): ($\\mu$) The mean value to which the noise gravitates
            theta (float, optional): ($\\theta$) The speed of mean reversion
            sigma (float, optional): ($\\sigma$) The scale of the random component
            device (torch.device, optional): the device to perform computations on
        """
        if size <= 0:
            raise ValueError("'size' must be larger than 0!")

        if theta <= 0 or sigma <= 0:
            raise ValueError("'theta' and 'sigma' must be larger than 0!")

        self.mu = torch.full((size,), mu, dtype=torch.float32, device=device)
        self.theta = theta
        self.sigma = sigma
        self.device = device

        self.state = None

        self.reset()

    def reset(self) -> None:
        """Resets the noise process to the mean ($\\mu$) value."""
        self.state = deepcopy(self.mu)

    def sample(self) -> torch.Tensor:
        """
        Generates a new noise sample.

        Returns:
            sample (torch.Tensor): A noise sample of the same shape as $\\mu$.
        """
        t = torch.randn(self.state.shape, dtype=torch.float32, device=self.device)
        dx = self.theta * (self.mu - self.state) + self.sigma * t

        self.state += dx
        return self.state
