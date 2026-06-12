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

from typing import Self

from flax import struct

from velora.cli.disco.settings import DiscoDashboardSettings, DiscoParamsSettings
from velora.disco.config.metadata import RuleTrainerMetadata
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

    Embedding dimensions are computed dynamically based on `prediction_size`
    using the `create()` method.

    Parameters
    ----------
    prediction_size : int
        Size of the observation/action-conditioned prediction vectors `(y, z)`.
    obs_embed_dim : int (optional)
        Embedding output dimension for observation-conditioned predictions `(y)`
    action_embed_dim : int (optional)
        Embedding output dimension for action-conditional inputs `(z, q, pi)`
    scalar_embed_dim : int (optional)
        Embedding output dimension for scalar inputs `(rewards, discounts)`
    output_dim : int (optional)
        Total output dimension of the encoder
    """

    prediction_size: int

    obs_embed_dim: int
    action_embed_dim: int
    scalar_embed_dim: int

    output_dim: int

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
            Size of the observation/action-conditioned prediction vectors `(y, z)`.

        Returns
        -------
        config : DiscoEncoderSettings
            New settings with computed embedding dimensions
        """
        embed_dim = prediction_size // 2
        scalar_embed_dim = max(prediction_size // 8, 4)

        return cls(
            prediction_size=prediction_size,
            obs_embed_dim=embed_dim,
            action_embed_dim=embed_dim,
            scalar_embed_dim=scalar_embed_dim,  # Min of 4
            output_dim=embed_dim * 4 + scalar_embed_dim,
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
    """

    n_hidden: int = 128
    prediction_size: int = 128

    lr: float = 3e-4
    max_grad_norm: float = 1.0
    sparsity: float = 0.5

    def encoder_config(self) -> DiscoEncoderSettings:
        """
        Extracts encoder configuration from agent settings.

        Returns
        -------
        config : DiscoEncoderSettings
            Encoder configuration
        """
        return DiscoEncoderSettings.create(self.prediction_size)


@struct.dataclass(frozen=True)
class DiscoValueSettings:
    """
    Dataclass for `DiscoValueAgent` settings.

    Parameters
    ----------
    lr : float (optional)
        Learning rate. Default is `0.0003`
    max_grad_norm : float (optional)
        Maximum gradient norm for gradient clipping. Default is `1.0`
    gamma : float (optional)
        Discount factor for rewards. Default is `0.997`
    td_lambda : float (optional)
        The `λ` in TD(λ) / V-Trace. Used in computing advantage estimates
        and temporal difference (TD) targets. Default is `0.95`.

        When -
        - `λ=0.0` → Use only 1-step TD (immediate reward + bootstrap)
        - `λ=1.0` → Use full Monte Carlo return (entire episode)
        - `λ=0.95` → Blend of n-step returns (weighted toward longer horizons)
    """

    lr: float = 3e-4
    max_grad_norm: float = 1.0
    gamma: float = 0.997
    td_lambda: float = 0.95


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
    meta_grad_norm : float (optional)
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
        Number of trajectories per agent update (composed of on-policy +
        off-policy samples) retrieved from a buffer. Default is `20`
    n_updates : int (optional)
        Number of agent updates to backpropagate through for meta-gradient
        computation (sliding window size). Default is `20`
    seq_len : int (optional)
        Number of timesteps per trajectory (rollout size; `T`).
        Default is `20`
    tau : float (optional)
        Soft update coefficient for target network updates. Default is `0.995`
    replay_ratio : float (optional)
        Fraction of each per-update batch drawn from replay. The rest is
        the most-recently inserted fresh trajectories. Default is `0.9`
    replay_capacity : int (optional)
        Per-agent ring size (in trajectory slots) for the replay buffer.
        Default is `1000`
    """

    agent: PolicyAgentSettings = struct.field(default_factory=PolicyAgentSettings)
    disco_agent: DiscoAgentSettings = struct.field(default_factory=DiscoAgentSettings)
    disco_value: DiscoValueSettings = struct.field(default_factory=DiscoValueSettings)
    ema: EMASettings = struct.field(default_factory=EMASettings)

    loss_cost: LossCostSettings = struct.field(default_factory=LossCostSettings)
    run: RunSettings = struct.field(default_factory=RunSettings)

    meta_lr: float = 0.001
    meta_grad_norm: float = 1.0
    entropy_coef: float = 1e-2
    reg_scale: float = 1e-3
    kl_reg: float = 1e-2

    total_env_steps: int = 500_000_000
    n_updates: int = 20
    batch_size: int = 20
    seq_len: int = 20

    tau: float = 0.995

    replay_ratio: float = 0.9
    replay_capacity: int = 1000

    @property
    def n_fresh_per_update(self) -> int:
        """
        Fresh on-policy trajectories drawn from the most-recent slice of
        the replay buffer per agent update (`1 - replay_ratio` share of
        each batch, rounded, lower-bounded at 1).
        """
        return max(1, round(self.batch_size * (1.0 - self.replay_ratio)))

    @property
    def n_fresh_per_meta(self) -> int:
        """
        Fresh on-policy trajectories collected per agent per meta-step.

        DiscoRL mixes each agent-update batch as `replay_ratio` replay +
        `(1 - replay_ratio)` on-policy. Across `n_updates` inner updates,
        this is `n_updates * n_fresh_per_update`. The training collect
        must write exactly this many trajectories so the buffer's
        "fresh" slice is genuinely on-policy.
        """
        return self.n_updates * self.n_fresh_per_update

    @property
    def steps_per_meta(self) -> int:
        """
        Environment steps consumed per trainer per meta-step.

        Formula: `n_fresh_per_meta * T + 2T` (on-policy collect +
        validation rollout).
        """
        return self.n_fresh_per_meta * self.seq_len + self.seq_len * 2

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

    def console_config(
        self,
        metadata: RuleTrainerMetadata,
        params: DiscoParamsSettings,
        jit_compile: bool,
        cache_status: str,
    ) -> DiscoDashboardSettings:
        """
        Sets the configuration for the `DiscoConsoleDashboard`.

        Parameters
        ----------
        metadata : RuleTrainerMetadata
            Rule trainer metadata
        params : DiscoParamsSettings
            Parameter counts for each agent type
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
            meta_steps=metadata.n_meta_steps,
            n_updates=self.n_updates,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            n_agents=metadata.num_trainers,
            total_steps=self.total_env_steps,
            log_dir=str(self.run.log_dir),
            cp_dir=str(self.run.checkpoint_dir),
            env_categories=metadata.env_categories,
            params=params,
            complete_path=str(self.run.dirpath),
            jit_compile=jit_compile,
            cache_status=cache_status,
        )
