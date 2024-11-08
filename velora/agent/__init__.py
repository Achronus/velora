import torch.nn as nn
from torch.optim import Optimizer

from velora.agent.storage import Storage, ReplayBuffer, Rollouts

from pydantic import BaseModel, ConfigDict


__all__ = [
    "Storage",
    "ReplayBuffer",
    "Rollouts",
]


class Agent(BaseModel):
    """
    Contains the logic for an Reinforcement Learning (RL) agent handling it's behaviour and learning process.

    Args:
        model (torch.nn.Module):
        optimizer (torch.optim.Optimizer):
        loss (torch.nn.Loss):
        storage (velora.agent.Storage):
    """

    model: nn.Module
    optimizer: Optimizer
    loss: nn.Module
    storage: Storage

    model_config = ConfigDict(arbitrary_types_allowed=True)
