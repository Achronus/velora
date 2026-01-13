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

from pathlib import Path

import jax.numpy as jnp
from flax import struct

from velora.core.distributions import CategoricalBins


@struct.dataclass(frozen=True)
class AgentSettings:
    """
    Dataclass for `PolicyAgent` settings.

    Parameters
    ----------
    n_hidden : int
        Number of decision nodes for policy networks (inter + command nodes)
    prediction_size : int
        Size of the observation/action-conditioned prediction vectors (y, z)
    q_size : int
        Size of action-value prediction head. Controls the number of discrete bins
        for distributional Q-values (`n_atoms`)
    lr : float (optional)
        Learning rate for the agent's optimizer. Default is `0.0003`
    max_grad_norm : float (optional)
        Maximum gradient norm for gradient clipping. Default is `1.0`
    bin_resolution : float (optional)
        Target bin resolution (width) for distributional Q-values. Dynamically
        sets the range of Q-values that can be represented for `[min, max]` based
        on `q_size`. Smaller resolutions provide finer granularity for value
        predictions but reduce the representable range. Default is `0.4`
    sparsity_level : float (optional)
        Network connection sparsity between neurons used for the Liquid Neural
        Networks (LNNs). Default is `0.5`
    """

    n_hidden: int
    prediction_size: int
    q_size: int
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    bin_resolution: float = 0.4
    sparsity_level: float = 0.5

    def categorical_bins(self) -> CategoricalBins:
        """
        Computes the categorical bin values for distributional Q-values.

        Returns
        -------
        bins : CategoricalBins
            An object containing the categorical bin values
        """
        max_bin_value = (self.bin_resolution * (self.q_size - 1)) / 2.0
        support = jnp.linspace(-max_bin_value, max_bin_value, num=self.q_size)
        return CategoricalBins(
            support=support,
            min_value=-max_bin_value,
            max_value=max_bin_value,
        )


@struct.dataclass(frozen=True)
class MetaAgentSettings:
    """
    Dataclass for `MetaAgent` settings.
    """


@struct.dataclass(frozen=True)
class MetaValueSettings:
    """
    Dataclass for `MetaValueNetwork` settings.

    The meta-value function estimates advantages for computing meta-gradients
    during rule discovery. It is **NOT** the agent's value function - it's
    infrastructure for meta-optimization.

    Parameters
    ----------
    lr : float (optional)
        Learning rate. Default is `0.0003`
    max_grad_clip : float (optional)
        Maximum value gradients can be. Default is `1.0`
    gamma : float (optional)
        Discount factor for rewards. Default is `0.997`
    td_lambda : float (optional)
        The `λ` in TD(λ) / V-Trace. Used in computing advantage estimates
        and temporal difference (TD) targets. Default is `0.95`.

        When -
        - `λ=0.0` → Use only 1-step TD (immediate reward + bootstrap)
        - `λ=1.0` → Use full Monte Carlo return (entire episode)
        - `λ=0.95` → Blend of n-step returns (weighted toward longer horizons)

    loss_weight : float (optional)
        The loss weight. Controls how much the meta-value function loss contributes
        to the total loss. Default is `1.0`
    ema_decay : float (optional)
        Exponential moving average (EMA) decay rate for advantage normalization.
        Default is `0.99`
    ema_eps : float (optional)
        Exponential moving average (EMA) epsilon for numerical stability.
        Default is `0.000001`
    """

    lr: float = 3e-4
    max_grad_clip: float = 1.0  # max_abs_update
    gamma: float = 0.997
    td_lambda: float = 0.95
    loss_weight: float = 1.0  # outer_value_cost
    ema_decay: float = 0.99
    ema_eps: float = 1e-6


@struct.dataclass(frozen=True)
class MixedBufferSettings:
    """
    Dataclass for `MixedBuffer` settings.

    Parameters
    ----------
    seq_len : int (optional)
        Number of timesteps per trajectory (rollout size; `T`).
        Default is `29`
    capacity : int (optional)
        Maximum number of trajectories to store (`N`).
        Default is `1024`
    split_ratio : float (optional)
        Fraction of samples from replay vs rollouts.
        Default is `0.9` (90% replay, 10% rollout)
    """

    seq_len: int = 29
    capacity: int = 1024
    split_ratio: float = 0.9


@struct.dataclass(frozen=True)
class LossCostSettings:
    """
    Dataclass for loss cost settings used during meta-optimization.

    These are static multipliers for each loss component, used for stability
    during training. All values are fixed at `1.0` except `value` which is `0.2`
    as per the original DiscoRL paper.

    Parameters
    ----------
    pi : float (optional)
        Weight for policy target loss ($KL(\\hat{\\pi}, \\pi)$).
        Default is `1.0`
    y : float (optional)
        Weight for observation-conditioned prediction loss ($KL(\\hat{y}, y)$). Default is `1.0`
    z : float (optional)
        Weight for action-conditioned prediction loss ($KL(\\hat{z}, z)$). Default is `1.0`
    value : float (optional)
        Weight for policy agents value function loss. Lower than other costs to
        prevent value loss from dominating. Default is `0.2`
    aux_pi : float (optional)
        Weight for auxiliary policy prediction loss (1-step policy prediction; $KL(\\hat{p}, p)$).
        Default is `1.0`
    """

    pi: float = 1.0
    y: float = 1.0
    z: float = 1.0
    value: float = 0.2
    aux_pi: float = 1.0


@struct.dataclass(frozen=True)
class AgentTrainerSettings:
    """
    Dataclass for `AgentTrainer` settings.

    Parameters
    ----------
    agent : AgentSettings
        Configuration for `PolicyAgent` architecture
    buffer : MixedBufferSettings
        Configuration for trajectory buffer
    num_envs : int
        Number of vectorized environments for throughput
    n_updates : int
        Number of agent updates to backpropagate through
    tau : float
        EMA coefficient for target network updates
    batch_size : int
        Number of trajectories per training batch
    checkpoint_dir : pathlib.Path
        Directory for saving/loading agent states
    """

    agent: AgentSettings
    buffer: MixedBufferSettings

    num_envs: int
    n_updates: int
    tau: float
    batch_size: int

    checkpoint_dir: Path


@struct.dataclass(frozen=True)
class RuleTrainerSettings:
    """
    Dataclass for `RuleTrainer` settings.

    Configures the outer loop of DiscoRL: managing a population of agents
    across environments, computing meta-gradients, and updating the `MetaAgent`
    to discover a learning rule.

    Parameters
    ----------
    agent : AgentSettings
        Configuration for `PolicyAgent` architecture
    meta_agent : MetaAgentSettings
        Configuration for `MetaAgent` architecture
    meta_value : MetaValueSettings (optional)
        Configuration for meta-value function used in meta-gradient computation.
        Default is `MetaValueSettings()`
    buffer : MixedBufferSettings (optional)
        Configuration for trajectory buffer. Default is `MixedBufferSettings()`
    loss_cost : LossCostSettings (optional)
        Loss component weights. Default is `LossCostSettings()`
    meta_lr : float (optional)
        Learning rate for meta-network optimizer. Default is `0.001`
    meta_grad_clip : float (optional)
        Gradient clipping threshold for meta-updates. Default is `1.0`
    n_steps : int (optional)
        Total number of meta-training steps. Default is `1_000_000`
    n_updates : int (optional)
        Number of agent updates to backpropagate through for meta-gradient
        computation (sliding window size). Default is `20`
    num_envs : int (optional)
        Number of vectorized environments. Default is `8`
    batch_size : int (optional)
        Number of trajectories per training batch. Default is `96`
    tau : float (optional)
        EMA coefficient for target network updates. Default is `0.9`
    checkpoint_dir : pathlib.Path (optional)
        Directory for saving agent states and checkpoints.
        Default is `./checkpoints`
    checkpoint_freq : int (optional)
        Meta-steps between full checkpoints. Default is `100`
    """

    agent: AgentSettings
    meta_agent: MetaAgentSettings
    meta_value: MetaValueSettings = MetaValueSettings()
    buffer: MixedBufferSettings = MixedBufferSettings()
    loss_cost: LossCostSettings = LossCostSettings()

    meta_lr: float = 0.001
    meta_grad_clip: float = 1.0
    n_steps: int = 1_000_000
    n_updates: int = 20

    num_envs: int = 8
    batch_size: int = 96
    tau: float = 0.9

    checkpoint_dir: Path = Path(".", "checkpoints")
    checkpoint_freq: int = 100

    def agent_trainer_config(self) -> AgentTrainerSettings:
        """
        Sets the configuration for the `AgentTrainer`.

        Returns
        -------
        config : AgentTrainerSettings
            Configuration for the `AgentTrainer`
        """
        return AgentTrainerSettings(
            agent=self.agent,
            buffer=self.buffer,
            num_envs=self.num_envs,
            n_updates=self.n_updates,
            tau=self.tau,
            batch_size=self.batch_size,
            checkpoint_dir=self.checkpoint_dir,
        )
