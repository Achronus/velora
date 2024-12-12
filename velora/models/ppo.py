from pydantic import BaseModel, ConfigDict, validate_call
from velora.models.base import AgentModel

import torch
import torch.nn as nn


class PPOInputs(BaseModel):
    in_channels: int
    n_actions: int
    input_dim: tuple[int, int]


class PPO(AgentModel):
    """
    A Proximal Policy Optimization (PPO) agent with a CNN backbone.

    Args:
        in_channels (int): The number of colour channels
        n_actions (int): the number of agent actions
        input_dim (tuple[int, int], optional): the size of the input image `(H, W)`
    """

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True),
        validate_return=True,
    )
    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        input_dim: tuple[int, int] = (84, 84),
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.feature_dim = self._calc_feature_dim(in_channels, input_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
        )
        self.critic = nn.Linear(256, 1)  # Policy mean
        self.actor = nn.Linear(256, n_actions)  # Value function
        self.log_std = nn.Parameter(torch.zeros(n_actions))

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True),
        validate_return=True,
    )
    def _calc_feature_dim(self, in_channels: int, input_dim: tuple[int, int]) -> int:
        """
        Calculates the size of the convolutional networks output and returns it.
        """
        with torch.no_grad():
            x = torch.zeros(1, in_channels, *input_dim)  # Dummy input
            return self.conv(x).size(1)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True),
        validate_return=True,
    )
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc(self.conv(x))
        mean = self.critic(x)
        value = self.actor(x)
        return mean, value, self.log_std.exp()

    def act(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.forward(x))

    def get_vf(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.forward(x))
