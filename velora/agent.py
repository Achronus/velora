import torch
import torch.nn as nn

from velora.storage import Storage

from pydantic import BaseModel


class Agent(BaseModel):
    """
    Contains the logic for an Reinforcement Learning (RL) agent handling it's behaviour and learning process.

    Args:
        model (torch.nn.Module):
        optimizer (torch.optim):
        loss (torch.nn.Loss):
        storage (velora.storage.Storage):
    """

    model: nn.Module
    optimizer: torch.optim
    loss: nn.Module
    storage: Storage
