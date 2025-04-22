from dataclasses import dataclass

import torch


@dataclass
class BatchExperience:
    """
    Storage container for a batch agent experiences.

    Attributes:
        states (torch.Tensor): a batch of environment observations
        actions (torch.Tensor): a batch of agent actions taken in the states
        rewards (torch.Tensor): a batch of rewards obtained for taking the actions
        next_states (torch.Tensor): a batch of newly generated environment
            observations following the actions taken
        dones (torch.Tensor): a batch of environment completion statuses
        hiddens (torch.Tensor): a batch of prediction network hidden states
            (e.g., Actor)
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    hiddens: torch.Tensor
