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
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


@dataclass
class RolloutBatchExperience(BatchExperience):
    """
    Storage container for a batch agent experiences.

    Attributes:
        states (torch.Tensor): a batch of environment observations
        actions (torch.Tensor): a batch of agent actions taken in the states
        rewards (torch.Tensor): a batch of rewards obtained for taking the actions
        next_states (torch.Tensor): a batch of newly generated environment
            observations following the actions taken
        dones (torch.Tensor): a batch of environment completion statuses
        log_probs (torch.Tensor): a batch of log probabilities of the actions
        values (torch.Tensor): a batch of state values
    """

    log_probs: torch.Tensor
    values: torch.Tensor
