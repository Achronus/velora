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
from typing import Dict, Self

from flax import struct

from velora.cli.disco.settings import DiscoDashboardSettings, DiscoParamsSettings
from velora.tracking.settings import RunSettings


@struct.dataclass(frozen=True)
class PolicyAgentSettings:
    """
    Dataclass for `PolicyAgent` settings.

    Parameters
    ----------
    n_hidden : int (optional)
        Number of decision nodes for policy networks (inter + command nodes).
        Default is `64`
    prediction_size : int (optional)
        Size of the observation/action-conditioned prediction vectors (y, z).
        Must match `DiscoAgent` size. Default is `128`
    min_log_std : float (optional)
        Minimum log standard deviation for
        the Gaussian policy. Prevents the policy from becoming too deterministic,
        which would cause gradient explosion in the Gaussian KL divergence.
        Default is `-5.0`
    max_log_std : float (optional)
        Maximum log standard deviation for the
        Gaussian policy. Prevents excessively noisy policies early in training.
        Default is `2.0`
    lr : float (optional)
        Learning rate for the agent's optimizer. Default is `0.0003`
    max_grad_norm : float (optional)
        Maximum gradient norm for gradient clipping. Default is `1.0`
    sparsity : float (optional)
        Network connection sparsity between neurons used for the Liquid Neural
        Networks (LNNs). Default is `0.5`.

        Must be a value between `[0.1, 0.9]`:

            - Where `0.1` neurons are very dense
            - Where `0.9` neurons are very sparse
    """

    n_hidden: int = 64
    prediction_size: int = 128

    min_log_std: float = -5.0
    max_log_std: float = 2.0

    lr: float = 3e-4
    max_grad_norm: float = 1.0
    sparsity: float = 0.5


@struct.dataclass(frozen=True)
class DiscoEncoderSettings:
    """
    Dataclass for `DiscoInputEncoder` settings.

    Embedding dimensions can be set manually or computed dynamically based on
    `prediction_size` using the `create()` method.

    The encoder derives `n_actions` from input shapes at runtime, enabling
    action-agnostic encoding that works across different action spaces.

    Parameters
    ----------
    prediction_size : int
        Size of the observation/action-conditioned prediction vectors `(y, z)`.
    obs_embed_dim : int (optional)
        Embedding output dimension for observation-conditioned predictions `(y)`.
        Default is `32`
    action_embed_dim : int (optional)
        Embedding output dimension for action-conditional inputs `(z, q, pi)`.
        Default is `32`
    scalar_embed_dim : int (optional)
        Embedding output dimension for scalar inputs `(rewards, discounts)`.
        Default is `8`
    """

    prediction_size: int

    obs_embed_dim: int = 32
    action_embed_dim: int = 32
    scalar_embed_dim: int = 16

    @classmethod
    def create(cls, prediction_size: int) -> Self:
        """
        Creates a new settings object using dynamic computation of embedding dimensions based on `prediction_size`.

        Uses the following ratios:
            - `obs_embed_dim` = `prediction_size // 2`
            - `action_embed_dim` = `prediction_size // 2`
            - `scalar_embed_dim` = `max(prediction_size // 8, 4)`

        Parameters
        ----------
        prediction_size : int
            Size of the observation/action-conditioned prediction vectors (y, z).

        Returns
        -------
        config : DiscoEncoderSettings
            New settings with computed embedding dimensions
        """
        return cls(
            prediction_size=prediction_size,
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


@struct.dataclass(frozen=True)
class DiscoAgentSettings:
    """
    Dataclass for `DiscoAgent` settings.

    The agent derives `n_actions` from input shapes at runtime, enabling
    action-agnostic target generation that works across different action spaces.

    Parameters
    ----------
    n_hidden : int (optional)
        Number of decision nodes for networks (inter + command nodes).
        Default is `128`
    prediction_size : int (optional)
        Size of the observation/action-conditioned prediction vectors (y, z).
        Must match `PolicyAgent` size. Default is `128`
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
        Manual embedding dimension for state-conditional predictions `(y)`.
        Only used when `dynamic_embed_dims=False`. Default is `64`
    action_embed_dim : int (optional)
        Manual embedding dimension for action-conditional inputs `(z, q, pi)`.
        Only used when `dynamic_embed_dims=False`. Default is `64`
    scalar_embed_dim : int (optional)
        Manual embedding dimension for scalars `(rewards, discounts)`.
        Only used when `dynamic_embed_dims=False`. Default is `16`
    max_action_dim : int (optional)
        Maximum continuous action dimensionality across all environments in
        the training set. Determined at runtime by probing environment action
        spaces — not a tunable hyperparameter.

        Used by `DiscoNetwork` to size the policy target projections
        `(μ̂, log σ̂)` and by `DiscoInputEncoder` to size the policy
        and action-conditional encoder inputs. Serialized with the config so
        that `load()` can reconstruct the correct network shapes.

        Set to `0` for discrete action spaces (unused). Default is `0`
    """

    n_hidden: int = 128
    prediction_size: int = 128

    lr: float = 3e-4
    max_grad_norm: float = 1.0
    sparsity: float = 0.5

    dynamic_embed_dims: bool = True
    obs_embed_dim: int = 64
    action_embed_dim: int = 64
    scalar_embed_dim: int = 16

    max_action_dim: int = 0

    def with_max_action_dim(self, max_action_dim: int) -> Self:
        """
        Return a copy with `max_action_dim` set.

        Parameters
        ----------
        max_action_dim : int
            Maximum continuous action dimensionality

        Returns
        -------
        config : DiscoAgentSettings
            Updated configuration
        """
        return self.__replace__(max_action_dim=max_action_dim)

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
            return DiscoEncoderSettings.create(self.prediction_size)

        return DiscoEncoderSettings(
            prediction_size=self.prediction_size,
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
        Default is `0.99`
    eps : float (optional)
        Epsilon for numerical stability when normalizing by EMA
        variance during policy agent training. Default is `1e-6`
    root_eps : float (optional)
        Epsilon value for standard deviation. Default is `1e-12`
    """

    decay: float = 0.99
    eps: float = 1e-6
    root_eps: float = 1e-12


@struct.dataclass(frozen=True)
class AgentTrainerSettings:
    """
    Dataclass for `AgentTrainer` settings.

    Parameters
    ----------
    agent : PolicyAgentSettings
        Configuration for `PolicyAgent` architecture
    value : DiscoValueSettings
        Configuration for the Disco value function
    ema : EMASettings
        Configuration for Exponential Moving Averages (EMAs)
    loss_cost : LossCostSettings
        Loss component weights
    tau : float
        EMA coefficient for target network updates
    seq_len : int
        Number of timesteps per trajectory (rollout size; `T`)
    n_updates : int
        Number of agent updates to backpropagate through for meta-gradient
        computation (sliding window size)
    batch_size : int
        Trajectory batch size (number of vectorized environments)
    """

    agent: PolicyAgentSettings
    value: DiscoValueSettings
    ema: EMASettings
    loss_costs: LossCostSettings

    tau: float
    seq_len: int
    n_updates: int
    batch_size: int


@struct.dataclass(frozen=True)
class RuleTrainerSettings:
    """
    Dataclass for `RuleTrainer` settings.

    Configures the outer loop of DiscoRL: managing a population of agents
    across environments, computing meta-gradients, and updating the target agent
    to discover a learning rule.

    Parameters
    ----------
    agent : PolicyAgentSettings (optional)
        Configuration for `PolicyAgent` architecture. Default is `PolicyAgentSettings()`
    disco_agent : DiscoAgentSettings (optional)
        Configuration for `DiscoAgent` architecture. Default is `DiscoAgentSettings()`
    disco_value : DiscoValueSettings (optional)
        Configuration for meta-value function used in meta-gradient computation.
        Default is `DiscoValueSettings()`
    ema : EMASettings (optional)
        Configuration for Exponential Moving Averages (EMAs). Default is `EMASettings()`
    loss_cost : LossCostSettings (optional)
        Loss component weights. Default is `LossCostSettings()`
    run : RunSettings (optional)
        Configuration for the run directory `(checkpoints, logs, metadata)`.
        Default is `RunSettings()`
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
    total_env_steps : int (optional)
        Total environment step budget across all trainers. Default is `500M`
    batch_size : int (optional)
        Trajectory batch size. Controls the number of vectorized environments used per agent.
        Default is `20`
    n_updates : int (optional)
        Number of agent updates to backpropagate through for meta-gradient
        computation (sliding window size). Default is `20`
    seq_len : int (optional)
        Number of timesteps per trajectory (rollout size; `T`).
        Default is `20`
    tau : float (optional)
        Soft update coefficient for target network updates. Default is `0.995`
    """

    agent: PolicyAgentSettings = struct.field(default_factory=PolicyAgentSettings)
    disco_agent: DiscoAgentSettings = struct.field(default_factory=DiscoAgentSettings)
    disco_value: DiscoValueSettings = struct.field(default_factory=DiscoValueSettings)
    ema: EMASettings = struct.field(default_factory=EMASettings)

    loss_cost: LossCostSettings = struct.field(default_factory=LossCostSettings)
    run: RunSettings = struct.field(default_factory=RunSettings)

    meta_lr: float = 0.001
    meta_grad_clip: float = 1.0
    entropy_coef: float = 1e-2
    reg_scale: float = 1e-3
    kl_reg: float = 1e-2

    total_env_steps: int = 500_000_000
    n_updates: int = 20
    batch_size: int = 20
    seq_len: int = 20

    tau: float = 0.995

    @property
    def steps_per_meta(self) -> int:
        """
        Environment steps consumed per trainer per meta-step.

        Formula: `N * T + 2T` (inner loop + validation rollout).
        """
        return self.n_updates * self.seq_len + self.seq_len * 2

    def n_meta_steps(self, num_trainers: int) -> int:
        """
        Computes the number of meta-steps to run.

        Formula: `n_steps = total_env_steps // (num_trainers * steps_per_meta)`

        Parameters
        ----------
        num_trainers : int
            Total number of active trainer slots

        Returns
        -------
        meta_steps : int
            Total number of meta-steps to run
        """
        return max(1, self.total_env_steps // (num_trainers * self.steps_per_meta))

    def estimate_total_env_steps(self, n_steps: int, num_trainers: int) -> int:
        """
        Computes a total environment step count budget, given a population size.

        Useful for estimating how much experience a run will consume before
        committing to it:
            `total_env_steps = n_steps * num_trainers * steps_per_meta`

        Parameters
        ----------
        n_steps : int
            Number of meta-steps
        num_trainers : int
            Total number of active trainer slots

        Returns
        -------
        total_env_steps : int
            Total environment steps that will be consumed
        """
        return n_steps * num_trainers * self.steps_per_meta

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
            tau=self.tau,
            seq_len=self.seq_len,
            n_updates=self.n_updates,
            batch_size=self.batch_size,
        )

    def console_config(
        self,
        envs: Dict[str, int],
        num_trainers: int,
        n_chunks: int,
        params: DiscoParamsSettings,
        complete_path: str,
        jit_compile: bool,
        cache_status: str,
    ) -> DiscoDashboardSettings:
        """
        Sets the configuration for the `DiscoConsoleDashboard`.

        Parameters
        ----------
        envs : Dict[str, int]
            Mapping of environment category names to counts
        num_trainers : int
            Number of agent trainers
        n_chunks : int
            Number of trainer chunks per meta-step
        params : DiscoParamsSettings
            Parameter counts for each agent type
        complete_path : str
            Checkpoint completion path
        jit_compile : bool
            Whether JIT compilation is enabled
        cache_status : str
            JIT compilation cache status

        Returns
        -------
        config : DiscoDashboardSettings
            Configuration for `DiscoConsoleDashboard`
        """
        return DiscoDashboardSettings(
            meta_steps=self.n_meta_steps(num_trainers),
            n_updates=self.n_updates,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            n_agents=num_trainers,
            total_steps=self.total_env_steps,
            n_chunks=n_chunks,
            log_dir=str(self.run.log_dir),
            cp_dir=str(self.run.checkpoint_dir),
            env_categories=envs,
            params=params,
            complete_path=complete_path,
            jit_compile=jit_compile,
            cache_status=cache_status,
        )
