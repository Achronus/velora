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

import json
from dataclasses import fields
from typing import Self

import jax.numpy as jnp
from flax import struct

from velora.disco.distributions import CategoricalBins
from velora.tracking.settings import CheckpointSettings, MetricLoggerSettings


@struct.dataclass(frozen=True)
class PolicyAgentSettings:
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
    sparsity : float (optional)
        Network connection sparsity between neurons used for the Liquid Neural
        Networks (LNNs). Default is `0.5`.

        Must be a value between `[0.1, 0.9]`:

            - Where `0.1` neurons are very dense
            - Where `0.9` neurons are very sparse
    """

    n_hidden: int
    prediction_size: int
    q_size: int

    lr: float = 3e-4
    max_grad_norm: float = 1.0
    bin_resolution: float = 0.4
    sparsity: float = 0.5

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
class DiscoEncoderSettings:
    """
    Dataclass for `DiscoInputEncoder` settings.

    Embedding dimensions can be set manually or computed dynamically based on
    `prediction_size` using the `create()` method.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions in the action space
    prediction_size : int
        Size of the observation/action-conditioned prediction vectors (y, z).
    q_size : int
        Size of action-value prediction head.
    obs_embed_dim : int (optional)
        Embedding output dimension for observation-conditioned predictions (y).
        Default is `32`
    action_embed_dim : int (optional)
        Embedding output dimension for action-conditional inputs (z, q, pi).
        Default is `32`
    scalar_embed_dim : int (optional)
        Embedding output dimension for scalar inputs (rewards, discounts).
        Default is `8`
    """

    n_actions: int
    prediction_size: int
    q_size: int

    obs_embed_dim: int = 32
    action_embed_dim: int = 32
    scalar_embed_dim: int = 16

    @classmethod
    def create(cls, n_actions: int, prediction_size: int, q_size: int) -> Self:
        """
        Creates a new settings object using dynamic computation of embedding dimensions based on `prediction_size`.

        Uses the following ratios:
            - `obs_embed_dim` = `prediction_size // 2`
            - `action_embed_dim` = `prediction_size // 2`
            - `scalar_embed_dim` = `max(prediction_size // 8, 4)`

        Parameters
        ----------
        n_actions : int
            Number of discrete actions in the action space
        prediction_size : int
            Size of the observation/action-conditioned prediction vectors (y, z).
        q_size : int
            Size of action-value prediction head.

        Returns
        -------
        config : DiscoEncoderSettings
            New settings with computed embedding dimensions
        """
        return cls(
            n_actions=n_actions,
            prediction_size=prediction_size,
            q_size=q_size,
            obs_embed_dim=prediction_size // 2,
            action_embed_dim=prediction_size // 2,
            scalar_embed_dim=max(prediction_size // 8, 4),  # Min of 4
        )

    @property
    def output_dim(self) -> int:
        """
        Total output dimension of the encoder. Used as input to Disco network.

        Returns
        -------
        dim : int
            Flattened embedding dimension
        """
        return (
            self.obs_embed_dim * 2 + self.action_embed_dim * 2 + self.scalar_embed_dim
        )

    @property
    def action_input_dim(self) -> int:
        """
        Input dimension for action-conditional encoder.

        Combines `(z, q, pi, one_hot_actions)` for normal and target predictions.

        Result: `2 * (prediction_size + q_size + 1) + 1`

        Returns
        -------
        dim : int
            Action encoder input dimension
        """
        return 2 * (self.prediction_size + self.q_size + 1) + 1


@struct.dataclass(frozen=True)
class DiscoAgentSettings:
    """
    Dataclass for `DiscoAgent` settings.

    Parameters
    ----------
    n_hidden : int
        Number of decision nodes for networks (inter + command nodes)
    prediction_size : int
        Size of the observation/action-conditioned prediction vectors (y, z)
    q_size : int
        Size of action-value prediction head
    n_actions : int
        Number of discrete actions in the action space
    lr : float (optional)
        Learning rate for the agent's optimizer. Default is `0.0003`
    max_grad_norm : float (optional)
        Maximum gradient norm for gradient clipping. Default is `1.0`
    sparsity : float (optional)
        Network connection sparsity for LNNs. Default is `0.5`
    dynamic_embed_dims : bool (optional)
        Whether to dynamically compute encoder embedding dimensions based on
        `prediction_size`. When `False`, uses manual values. Default is `True`

        When `True` uses the following ratios:
            - `obs_embed_dim` = `prediction_size // 2`
            - `action_embed_dim` = `prediction_size // 2`
            - `scalar_embed_dim` = `max(prediction_size // 8, 4)`

    obs_embed_dim : int (optional)
        Manual embedding dimension for state-conditional predictions (y).
        Only used when `dynamic_embed_dims=False`. Default is `32`
    action_embed_dim : int (optional)
        Manual embedding dimension for action-conditional inputs (z, q, pi).
        Only used when `dynamic_embed_dims=False`. Default is `32`
    scalar_embed_dim : int (optional)
        Manual embedding dimension for scalars (rewards, discounts).
        Only used when `dynamic_embed_dims=False`. Default is `8`
    """

    n_hidden: int
    prediction_size: int
    q_size: int
    n_actions: int

    lr: float = 3e-4
    max_grad_norm: float = 1.0
    sparsity: float = 0.5

    dynamic_embed_dims: bool = True
    obs_embed_dim: int = 32
    action_embed_dim: int = 32
    scalar_embed_dim: int = 8

    def encoder_config(self) -> DiscoEncoderSettings:
        """
        Extracts encoder configuration from agent settings.

        When `dynamic_embed_dims=True`, computes embedding dimensions
        automatically. Otherwise, uses manual values.

        Returns
        -------
        config : DiscoEncoderSettings
            Encoder configuration
        """
        if self.dynamic_embed_dims:
            return DiscoEncoderSettings.create(
                self.n_actions,
                self.prediction_size,
                self.q_size,
            )

        return DiscoEncoderSettings(
            n_actions=self.n_actions,
            prediction_size=self.prediction_size,
            q_size=self.q_size,
            obs_embed_dim=self.obs_embed_dim,
            action_embed_dim=self.action_embed_dim,
            scalar_embed_dim=self.scalar_embed_dim,
        )

    def to_json(self, **extras) -> str:
        """
        Converts the object to a JSON string with `extras` added.

        Parameters
        ----------
        extras : kwargs (optional)
            Additional key-value arguments to merge into JSON string.
            Default is `None`

        Returns
        -------
        json : str
            The object as a JSON serialized string
        """
        items = {f.name: getattr(self, f.name) for f in fields(self)}
        return json.dumps(items | (extras or {}), indent=2)


@struct.dataclass(frozen=True)
class DiscoValueSettings:
    """
    Dataclass for `DiscoValueAgent` settings.

    Parameters
    ----------
    lr : float (optional)
        Learning rate. Default is `0.0003`
    max_grad_clip : float (optional)
        Maximum gradient clip value. Default is `1.0`
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
        Weight for value loss contribution to total meta-loss. Default is `1.0`
    """

    lr: float = 3e-4
    max_grad_clip: float = 1.0
    gamma: float = 0.997
    td_lambda: float = 0.95
    loss_weight: float = 1.0


@struct.dataclass(frozen=True)
class LossCostSettings:
    """
    Dataclass for loss cost settings used during meta-optimization of the target Meta-Network.

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
class EMASettings:
    """
    Dataclass for `EMAState` and method settings.

    Parameters
    ----------
    decay : float (optional)
        Exponential moving average (EMA) decay rate for normalizing
        advantages and TD errors during policy agent value learning.
        Higher values give more weight to historical statistics.
        Default is `0.999`
    eps : float (optional)
        Epsilon for numerical stability when normalizing by EMA
        variance during policy agent training. Default is `1e-6`
    root_eps : float (optional)
        Epsilon value for standard deviation. Default is `1e-12`
    """

    decay: float = 0.999
    eps: float = 1e-6
    root_eps: float = 1e-12


@struct.dataclass(frozen=True)
class AgentTrainerSettings:
    """
    Dataclass for `AgentTrainer` settings.

    Parameters
    ----------
    agent : AgentSettings
        Configuration for `PolicyAgent` architecture
    value : DiscoValueSettings
        Configuration for the Disco value function
    ema : EMASettings
        Configuration for Exponential Moving Averages (EMAs)
    loss_cost : LossCostSettings
        Loss component weights
    num_vec_envs : int
        Number of vectorized environments for throughput
    tau : float
        EMA coefficient for target network updates
    seq_len : int
        Number of timesteps per trajectory (rollout size; `T`)
    """

    agent: PolicyAgentSettings
    value: DiscoValueSettings
    ema: EMASettings
    loss_costs: LossCostSettings

    num_vec_envs: int
    tau: float
    seq_len: int


@struct.dataclass(frozen=True)
class RuleTrainerSettings:
    """
    Dataclass for `RuleTrainer` settings.

    Configures the outer loop of DiscoRL: managing a population of agents
    across environments, computing meta-gradients, and updating the target agent
    to discover a learning rule.

    Parameters
    ----------
    agent : AgentSettings
        Configuration for `PolicyAgent` architecture
    disco_agent : DiscoAgentSettings
        Configuration for `DiscoAgent` architecture
    disco_value : DiscoValueSettings (optional)
        Configuration for meta-value function used in meta-gradient computation.
        Default is `DiscoValueSettings()`
    ema : EMASettings (optional)
        Configuration for Exponential Moving Averages (EMAs). Default is `EMASettings()`
    loss_cost : LossCostSettings (optional)
        Loss component weights. Default is `LossCostSettings()`
    checkpoint : CheckpointSettings (optional)
        Configuration for checkpoints. Default is `CheckpointSettings()`
    logger : MetricLoggerSettings (optional)
        Configuration for `MetricsLogger`. Default is `MetricLoggerSettings()`
    meta_lr : float (optional)
        Learning rate for meta-network optimizer. Default is `0.001`
    meta_grad_clip : float (optional)
        Gradient clipping threshold for meta-updates. Default is `1.0`
    entropy_coef : float (optional)
        Coefficient for entropy regularization in the meta-loss. Encourages
        exploration by penalizing low-entropy (overly deterministic) policies.
        Higher values promote more exploration. Default is `0.01`
    reg_scale : float (optional)
        L2 regularization scale for meta-network target outputs. Penalizes
        large target means to prevent divergence. Default is `0.001`
    kl_reg : float (optional)
        KL divergence regularization scale between meta-network targets and
        current policy. Encourages stability by keeping targets close to the
        agent's current predictions. Default is `0.01`
    n_steps : int (optional)
        Total number of meta-training steps. Default is `1_000_000`
    n_updates : int (optional)
        Number of agent updates to backpropagate through for meta-gradient
        computation (sliding window size). Default is `20`
    seq_len : int (optional)
        Number of timesteps per trajectory (rollout size; `T`).
        Default is `29`
    num_vec_envs : int (optional)
        Number of vectorized environments. Default is `8`
    batch_size : int (optional)
        Number of trajectories per training batch. Default is `96`
    tau : float (optional)
        Soft update coefficient for target network updates. Default is `0.9`
    """

    agent: PolicyAgentSettings
    disco_agent: DiscoAgentSettings
    disco_value: DiscoValueSettings = DiscoValueSettings()
    ema: EMASettings = EMASettings()

    loss_cost: LossCostSettings = LossCostSettings()
    checkpoint: CheckpointSettings = CheckpointSettings()
    logger: MetricLoggerSettings = MetricLoggerSettings()

    meta_lr: float = 0.001
    meta_grad_clip: float = 1.0
    entropy_coef: float = 1e-2
    reg_scale: float = 1e-3
    kl_reg: float = 1e-2

    n_steps: int = 1_000_000
    n_updates: int = 20
    seq_len: int = 29

    num_vec_envs: int = 8
    batch_size: int = 96
    tau: float = 0.9

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
            value=self.disco_value,
            ema=self.ema,
            loss_costs=self.loss_cost,
            num_vec_envs=self.num_vec_envs,
            tau=self.tau,
            seq_len=self.seq_len,
        )
