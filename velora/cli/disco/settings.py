# Copyright 2025 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

from velora.base.outputs import ParamCount

if TYPE_CHECKING:
    from velora.disco.inputs import ActionGroup


@dataclass
class DiscoLosses:
    """
    Dataclass for DiscoRL training losses.

    Parameters
    ----------
    meta : float (optional)
        Meta training loss. Default is `0.0`
    policy_gradient : float (optional)
        Policy gradient loss. Default is `0.0`
    entropy : float (optional)
        Entropy loss. Default is `0.0`
    regularization : float (optional)
        Regularization loss. Default is `0.0`
    gradient_norm : float (optional)
        Gradient normalization. Default is `0.0`
    """

    meta: float = 0.0
    policy_gradient: float = 0.0
    entropy: float = 0.0
    regularization: float = 0.0
    gradient_norm: float = 0.0


@dataclass
class DiscoStats:
    """
    Dataclass for DiscoRL training statistics.

    Parameters
    ----------
    avg_reward : float (optional)
        Average reward. Default is `0.0`
    avg_length : float (optional)
        Average episode length. Default is `0.0`
    reward_std : float (optional)
        Reward standard deviation. Default is `0.0`
    reward_min : float (optional)
        Minimum reward. Default is `0.0`
    reward_max : float (optional)
        Maximum reward. Default is `0.0`
    """

    avg_reward: float = 0.0
    avg_length: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0


@dataclass
class DiscoParamsSettings:
    """
    Parameter counts for DiscoRL agents.

    Parameters
    ----------
    policy : ParamCount
        Active and total parameters for the policy agent
    value : ParamCount
        Active and total parameters for the value agent
    disco : ParamCount
        Active and total parameters for the disco meta-network
    """

    policy: ParamCount
    value: ParamCount
    disco: ParamCount


@dataclass
class DiscoDashboardSettings:
    """
    Configuration settings for the `DiscoConsoleDashboard`.

    Parameters
    ----------
    meta_steps : int
        Number of meta-training steps per environment
    n_updates : int
        Number of inner loop agent updates per meta-step
    seq_len : int
        Trajectory sequence length (timesteps per rollout)
    n_agents : int
        Number of agents in the population
    total_steps : int
        Total number of training steps
    action_groups : List[ActionGroup]
        A list of action groups for the trainer population
    log_dir : str
        Directory path for Tensorboard logs
    cp_dir : str
        Directory path for checkpoints
    env_categories : Dict[str, int]
        Mapping of environment category names to counts
    params : DiscoParamsSettings
        Parameter counts for each agent type
    complete_path : str
        Checkpoint completion path
    jit_compile : bool
        Whether JIT compilation is enabled
    cache_status : str
        JIT compilation cache status
    """

    meta_steps: int
    n_updates: int
    seq_len: int
    n_agents: int
    total_steps: int
    action_groups: List["ActionGroup"]

    log_dir: str
    cp_dir: str

    env_categories: Dict[str, int]

    params: DiscoParamsSettings
    complete_path: str
    jit_compile: bool
    cache_status: str

    def env_total(self) -> int:
        """
        Computes the total number of training environments.

        Returns
        -------
        total : int
            Environment training total
        """
        return sum(self.env_categories.values())

    def n_action_groups(self) -> int:
        """
        Computes the total number of action groups.

        Returns
        -------
        count : int
            Number of action groups
        """
        return len(self.action_groups)
