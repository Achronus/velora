from dataclasses import astuple, dataclass
from typing import Tuple

import torch


@dataclass
class Experience:
    """
    Storage container for a single agent experience.

    Parameters:
        state (torch.Tensor): an environment observation
        action (torch.Tensor): agent actions taken in the state
        reward (float): reward obtained for taking the action
        next_state (torch.Tensor): a newly generated environment observation
            after performing the action
        done (bool): environment completion status
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool

    def __iter__(self) -> Tuple:
        """
        Iteratively unpacks experience instances as tuples.

        Best used with the [`zip()`](https://docs.python.org/3/library/functions.html#zip) method:

        ```python
        batch = [Experience(...), Experience(...), Experience(...)]
        states, actions, rewards, next_states, dones = zip(*batch)

        # ((s1, s2, s3), (a1, a2, a3), (r1, r2, r3), (ns1, ns2, ns3), (d1, d2, d3))
        ```

        Returns:
            exp (Tuple): the experience as a tuple in the form `(state, action, reward, next_state, done)`.
        """
        return iter(astuple(self))


@dataclass
class BatchExperience:
    """
    Storage container for a batch agent experiences.

    Parameters:
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
