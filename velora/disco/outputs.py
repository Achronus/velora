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

from dataclasses import field, fields
from typing import Dict, Self, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from velora.disco.config.settings import LossCostSettings
from velora.disco.ema import EMAState
from velora.utils.transforms import squeeze_time


@struct.dataclass
class OCMPredictions:
    """
    Dataclass for `OCM` network predictions.

    Parameters
    ----------
    embedding : chex.Array
        Command layer output with shape `(B, T, F)`

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_features (`F`) - the number of features.

    pi : jax.Array
        Policy logits with shape `(B, T, A)` or `(B, A)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.
    y : jax.Array
        Observation-conditioned prediction vector with shape `(B, T, Y)` or `(B, Y)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - y_dim (`Y`) - the size of the observation-conditioned prediction vector.
    """

    embedding: chex.Array
    pi: chex.Array
    y: chex.Array

    def output_values(self, ignore_embed: bool = True) -> Tuple[chex.Array, ...]:
        """
        Convert object into a tuple of values.

        Parameters
        ----------
        ignore_embed : bool (optional)
            Flag for including the `embedding` field to the dict. Not added by default.
            Default is `True`

        Returns
        -------
        ocm_preds : Tuple[chex.Array, ...]
            OCM prediction values in order `(embedding, pi, y)`
        """
        skip = {"embedding"} if ignore_embed else set()
        return tuple(getattr(self, f.name) for f in fields(self) if f.name not in skip)


@struct.dataclass
class ACMPredictions:
    """
    Dataclass for `ACM` network predictions.

    Parameters
    ----------
    embedding : chex.Array
        Command layer output with shape `(B, T, F)`

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_features (`F`) - the number of features.
    z : jax.Array
        Action-conditioned prediction vector with shape `(B, T, A, Z)` or `(B, A, Z)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.
        - z_dim (`Z`) - the size of the action-conditioned prediction vector.

    aux_pi : jax.Array
        Auxiliary policy logits with shape `(B, T, A, A)` or `(B, A, A)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.

    q : jax.Array
        Action-value predictions with shape `(B, T, A, Q)` or `(B, A, Q)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.
        - q_dim (`Q`) - the size of the action-value prediction head.
    """

    embedding: chex.Array
    z: chex.Array
    aux_pi: chex.Array
    q: chex.Array

    def output_values(self, ignore_embed: bool = True) -> Tuple[chex.Array, ...]:
        """
        Convert object into a tuple of values.

        Parameters
        ----------
        ignore_embed : bool (optional)
            Flag for including the `embedding` field to the dict. Not added by default.
            Default is `True`

        Returns
        -------
        ocm_preds : Tuple[chex.Array, ...]
            OCM prediction values in order `(embedding, z, aux_pi, q)`
        """
        skip = {"embedding"} if ignore_embed else set()
        return tuple(getattr(self, f.name) for f in fields(self) if f.name not in skip)


@struct.dataclass
class DiscoPredictions:
    """
    Dataclass for `DiscoNetwork` predictions.

    Parameters
    ----------
    embedding : chex.Array
        Command layer output with shape `(B, T, F)`

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_features (`F`) - the number of features.

    pi : jax.Array
        Policy targets (`π̂,`) with shape `(B, T, A)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.
    y : jax.Array
        Observation-conditioned targets (`ŷ`) with shape `(B, T, Y)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - y_dim (`Y`) - the size of the observation-conditioned vector.
    z : jax.Array
        Action-conditioned targets (`ẑ`) with shape `(B, T, Z)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - z_dim (`Z`) - the size of the action-conditioned vector.
    """

    embedding: chex.Array
    pi: chex.Array
    y: chex.Array
    z: chex.Array

    def output_values(self, ignore_embed: bool = True) -> Tuple[chex.Array, ...]:
        """
        Convert object into a tuple of values.

        Parameters
        ----------
        ignore_embed : bool (optional)
            Flag for including the `embedding` field to the dict. Not added by default.
            Default is `True`

        Returns
        -------
        target_preds : Tuple[chex.Array, ...]
            Target prediction values in order `(embedding, pi, y, z)`
        """
        skip = {"embedding"} if ignore_embed else set()
        return tuple(getattr(self, f.name) for f in fields(self) if f.name not in skip)


@struct.dataclass
class PolicyAgentOutput:
    """
    Dataclass for the `PolicyAgent` output.

    If `T` dimension on values is `T=1` use `AgentOutput.create()`.

    Parameters
    ----------
    encoding : chex.Array
        Encoder embedding output `(B, T, F)`

        - `batch_size (B)` the number of samples per timestep
        - `seq_length (T)` the number of sequences (e.g., trajectories)
        - `features (F)` number of features in the embedding

    pi : jax.Array
        Policy logits with shape `(B, T, A)` or `(B, A)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.

    y : jax.Array
        Observation-conditioned prediction vector with shape `(B, T, Y)` or `(B, Y)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - y_dim (`Y`) - the size of the observation-conditioned prediction vector.

    z : jax.Array
        Action-conditioned prediction vector with shape `(B, T, A, Z)` or `(B, A, Z)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.
        - z_dim (`Z`) - the size of the action-conditioned prediction vector.

    aux_pi : jax.Array
        Auxiliary policy logits with shape `(B, T, A, A)` or `(B, A, A)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.

    q : jax.Array
        Action-value predictions with shape `(B, T, A, Q)` or `(B, A, Q)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.
        - q_dim (`Q`) - the size of the action-value prediction head.
    """

    encoding: chex.Array
    pi: chex.Array
    y: chex.Array
    z: chex.Array
    aux_pi: chex.Array
    q: chex.Array

    @classmethod
    def create(
        cls,
        encoding: chex.Array,
        pi: chex.Array,
        y: chex.Array,
        z: chex.Array,
        aux_pi: chex.Array,
        q: chex.Array,
    ) -> Self:
        """Create a new instance with time dimension squeezed if `T=1`."""
        return cls(*jax.tree.map(squeeze_time, (encoding, pi, y, z, aux_pi, q)))

    def to_numpy(self) -> Self:
        """Convert all fields to numpy arrays."""
        return jax.tree.map(np.asarray, self)

    @classmethod
    def from_numpy(
        cls,
        encoding: np.ndarray,
        pi: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        aux_pi: np.ndarray,
        q: np.ndarray,
    ) -> Self:
        """Create a new instance from numpy arrays."""
        return cls(*jax.tree.map(jnp.asarray, (encoding, pi, y, z, aux_pi, q)))


@struct.dataclass
class DiscoAgentOutput:
    """
    Dataclass for the `DiscoAgent` output.

    Parameters
    ----------
    pi : jax.Array
        Policy logits with shape `(B, T, A)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - n_actions (`A`) - the number of discrete actions in the action space.

    y : jax.Array
        Observation-conditioned prediction vector with shape `(B, T, Y)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - y_dim (`Y`) - the size of the observation-conditioned prediction vector.

    z : jax.Array
        Action-conditioned prediction vector with shape `(B, T, Z)`:

        - batch_size (`B`) - the number of samples per timestep.
        - seq_length (`T`) - the number of timesteps in the trajectory (can be more than one episode)
        - z_dim (`Z`) - the size of the action-conditioned prediction vector.
    """

    pi: chex.Array
    y: chex.Array
    z: chex.Array


@struct.dataclass
class ValueOutputs:
    """
    Outputs from value function computation.

    Parameters
    ----------
    value : chex.Array
        State-value estimates `V(s)`. Shape: `(T, B)`
    value_targets : chex.Array
        V-trace value targets. Shape: `(T-1, B)`
    advantages : chex.Array
        Advantage estimates `A(s, a)`. Shape: `(T-1, B)`
    normalized_advantages : chex.Array
        EMA-normalized advantages. Shape: `(T-1, B)`
    td : chex.Array
        Temporal difference errors. Shape: `(T-1, B)`
    normalized_td : chex.Array
        EMA-normalized TD errors. Shape: `(T-1, B)`
    rho : chex.Array
        Importance sampling weights. Shape: `(T-1, B)`
    """

    value: chex.Array
    value_targets: chex.Array
    advantages: chex.Array
    normalized_advantages: chex.Array
    td: chex.Array
    normalized_td: chex.Array
    rho: chex.Array

    def to_metrics(self) -> Dict[str, float]:
        """
        Convert to metrics dict for logging.

        Returns
        -------
        dict : Dict[str, float]
            Object in dictionary format
        """
        return {
            "value/mean": float(self.value.mean()),
            "value/advantage_mean": float(self.advantages.mean()),
            "value/advantage_std": float(self.advantages.std()),
            "value/normalized_advantage_mean": float(self.normalized_advantages.mean()),
            "value/normalized_advantage_std": float(self.normalized_advantages.std()),
            "value/td_mean": float(self.td.mean()),
            "value/td_std": float(self.td.std()),
            "value/normalized_td_mean": float(self.normalized_td.mean()),
            "value/rho_mean": float(self.rho.mean()),
        }


@struct.dataclass
class AgentLosses:
    """
    Dataclass for individual loss components from a policy agent training step.

    All losses are computed as KL divergences between agent predictions and targets generated by the meta-network, except for value (distributional TD) and aux (auxiliary policy prediction).

    Parameters
    ----------
    pi : chex.Array
        Policy loss. KL divergence between agent's policy logits and
        meta-network's target policy `π̂.`
    y : chex.Array
        Observation-conditioned prediction loss. KL divergence between
        agent's `y` predictions and meta-network's target `ŷ`.
    z : chex.Array
        Action-conditioned prediction loss. KL divergence between
        agent's `z` predictions and meta-network's target `ẑ`.
    aux_pi : chex.Array
        Auxiliary policy loss. Prediction error for the auxiliary policy head (predicts policy from action-conditioned representations).
    total : chex.Array (optional)
        Weighted sum of all loss components. Should be computed using the
        `compute_total()` method. Default is `jnp.array(0)` as placeholder
    """

    pi: chex.Array
    y: chex.Array
    z: chex.Array
    aux_pi: chex.Array
    total: chex.Array = field(default_factory=lambda: jnp.array(0))

    def compute_total(self, loss_costs: LossCostSettings) -> Self:
        """
        Compute the weighted sum of all loss components and update `total`.

        Parameters
        ----------
        loss_costs : LossCostSettings
            Weights for each loss component

        Returns
        -------
        self : AgentLosses
            New instance with updated `total` field
        """
        total = sum(
            getattr(loss_costs, f.name) * getattr(self, f.name)
            for f in fields(self)
            if f.name != "total"
        )
        return self.__replace__(total=jnp.array(total))

    def to_metrics(self) -> Dict[str, float]:
        """
        Convert to metrics dict for logging.

        Returns
        -------
        dict : Dict[str, float]
            Object in dictionary format
        """
        return {
            f"inner/{f.name}_loss": float(getattr(self, f.name)) for f in fields(self)
        }


@struct.dataclass
class AgentLossAux:
    """
    Dataclass for the `loss_fn` in `AgentTrainer`.

    Parameters
    ----------
    value_outs : ValueOutputs
        Value function outputs
    adv_ema : EMAState
        EMA state for advantage normalization
    td_ema : EMAState
        EMA state for TD normalization
    """

    value_outs: ValueOutputs
    adv_ema: EMAState
    td_ema: EMAState


@struct.dataclass
class MetaLossAux:
    """
    Dataclass for the `meta_loss_fn` in `RuleTrainer`.

    Parameters
    ----------
    pg_loss : chex.Array
        Policy gradient loss
    entropy_loss : chex.Array
        Entropy loss
    reg_loss : chex.Array
        Regularization loss
    disco_h : chex.Array
        Disco Network hidden state (`DiscoAgent`)
    meta_h : chex.Array
        Meta LNN hidden state (`DiscoAgent`)
    p_params : optax.Params
        Policy agent parameters
    value_outs : ValueOutputs
        Value function outputs
    v_params : optax.Params
        Value agent parameters
    v_opt_state : optax.OptState
        Value agent optimizer state
    """

    pg_loss: chex.Array
    entropy_loss: chex.Array
    reg_loss: chex.Array
    disco_h: chex.Array
    meta_h: chex.Array
    p_params: optax.Params
    value_outs: ValueOutputs
    v_params: optax.Params
    v_opt_state: optax.OptState


@struct.dataclass
class MetaInnerStepCarry:
    """
    Carry state for the `_inner_step` in `RuleTrainer._compute_meta_gradient`.

    Parameters
    ----------
    p_params : optax.Params
        Policy agent parameters
    v_params : optax.Params
        Value agent parameters
    disco_h : chex.Array
        Disco network hidden state
    meta_h : chex.Array
        Meta LNN hidden state
    p_opt_state : optax.OptState
        Policy agent optimizer state
    v_opt_state : optax.OptState
        Value agent optimizer state
    adv_ema : EMAState
        Advantage EMA state
    td_ema : EMAState
        TD-error EMA state
    """

    p_params: optax.Params
    v_params: optax.Params
    disco_h: chex.Array
    meta_h: chex.Array
    p_opt_state: optax.OptState
    v_opt_state: optax.OptState
    adv_ema: EMAState
    td_ema: EMAState


@struct.dataclass
class MetaGradOutput:
    """
    Outputs from a single meta-gradient computation.

    Parameters
    ----------
    meta_grad : ArrayTree
        Meta-gradient with respect to the DiscoAgent parameters
    disco_h : chex.Array
        Updated DiscoNetwork hidden state
    meta_h : chex.Array
        Updated Meta-LNN hidden state
    p_params : optax.Params
        Policy parameters after N inner updates
    v_params : optax.Params
        Value parameters after N inner updates
    v_opt_state : optax.OptState
        Value optimizer state after N inner updates
    pg_loss : chex.Array
        Policy gradient loss on the validation rollout
    entropy_loss : chex.Array
        Entropy regularisation loss
    reg_loss : chex.Array
        Meta-regularisation loss (KL + entropy on predictions)
    meta_loss : chex.Array
        Total meta-loss (pg + entropy + reg)
    advantages : chex.Array
        Raw advantages from the validation rollout
    normalized_advantages : chex.Array
        EMA-normalised advantages from the validation rollout
    adv_ema : EMAState
        Advantage EMA state after all N inner steps
    td_ema : EMAState
        TD-error EMA state after all N inner steps
    """

    meta_grad: chex.ArrayTree
    disco_h: chex.Array
    meta_h: chex.Array
    p_params: optax.Params
    v_params: optax.Params
    v_opt_state: optax.OptState
    pg_loss: chex.Array
    entropy_loss: chex.Array
    reg_loss: chex.Array
    meta_loss: chex.Array
    advantages: chex.Array
    normalized_advantages: chex.Array
    adv_ema: EMAState
    td_ema: EMAState


@struct.dataclass
class LossStatistics:
    """
    Loss statistics for meta-training.

    Parameters
    ----------
    meta : chex.Array
        Total meta loss
    policy_gradient : chex.Array
        Policy gradient loss
    entropy : chex.Array
        Entropy loss
    regularization : chex.Array
        Regularization loss
    """

    meta: chex.Array
    policy_gradient: chex.Array
    entropy: chex.Array
    regularization: chex.Array

    def to_dict(self) -> Dict[str, float]:
        return {
            "meta": float(self.meta),
            "policy_gradient": float(self.policy_gradient),
            "entropy": float(self.entropy),
            "regularization": float(self.regularization),
        }


@struct.dataclass
class RewardStatistics:
    """
    Per-trainer episodic rewards and lengths.

    Parameters
    ----------
    rewards : chex.Array
        Windowed mean returns across trainers `(N,)`
    lengths : chex.Array
        Windows mean episode lengths across trainers `(N,)`
    """

    rewards: chex.Array
    lengths: chex.Array

    @property
    def has_data(self) -> bool:
        """Check if at least one trainer completed an episode this window."""
        return jnp.shape(self.rewards)[0] > 0

    def summary(self, cat: str = "") -> Dict[str, float]:
        """
        Scalar summary statistics for logging.

        Computes `mean`, `std`, `min`, `max` for both rewards and lengths.
        Returns zeros for all fields if no episodes completed.

        Parameters
        ----------
        cat : str (optional)
            An optional category for the statistics. E.g., `"meta/"` =
            `"meta/avg_reward"`

        Returns
        -------
        stats : Dict[str, float]
            Keys - `[avg_reward, reward_std, reward_min, reward_max,avg_length, length_std, length_min, length_max]`
        """
        if not self.has_data:
            return {
                f"{cat}avg_reward": 0.0,
                f"{cat}reward_std": 0.0,
                f"{cat}reward_min": 0.0,
                f"{cat}reward_max": 0.0,
                f"{cat}avg_length": 0.0,
                f"{cat}length_std": 0.0,
                f"{cat}length_min": 0.0,
                f"{cat}length_max": 0.0,
            }

        return {
            f"{cat}avg_reward": float(jnp.mean(self.rewards)),
            f"{cat}reward_std": float(jnp.std(self.rewards)),
            f"{cat}reward_min": float(jnp.min(self.rewards)),
            f"{cat}reward_max": float(jnp.max(self.rewards)),
            f"{cat}avg_length": float(jnp.mean(self.lengths)),
            f"{cat}length_std": float(jnp.std(self.lengths)),
            f"{cat}length_min": float(jnp.min(self.lengths)),
            f"{cat}length_max": float(jnp.max(self.lengths)),
        }

    def core_stats(self) -> Dict[str, float]:
        """
        Core scalar reward statistics for logging.

        Returns
        -------
        stats : Dict[str, float]
            Keys - `[avg_reward, avg_length, reward_std, reward_min, reward_max]`
        """
        if not self.has_data:
            return {
                "avg_reward": 0.0,
                "avg_length": 0.0,
                "reward_std": 0.0,
                "reward_min": 0.0,
                "reward_max": 0.0,
            }

        return {
            "avg_reward": float(jnp.mean(self.rewards)),
            "avg_length": float(jnp.mean(self.lengths)),
            "reward_std": float(jnp.std(self.rewards)),
            "reward_min": float(jnp.min(self.rewards)),
            "reward_max": float(jnp.max(self.rewards)),
        }


class MetaStepStats:
    """
    Accumulator for per-trainer metrics across one meta-step.

    Parameters
    ----------
    num_trainers : int
        Number of agent trainers
    """

    def __init__(self, num_trainers: int) -> None:
        self._num_trainers = num_trainers
        self._idx = 0
        self._reward_count = 0

        zero = jnp.zeros((num_trainers,), dtype=jnp.float32)

        # Pre-allocate fixed-size buffers
        self._reward_buf: chex.Array = zero
        self._length_buf: chex.Array = zero

        self._losses = LossStatistics(
            meta=zero,
            policy_gradient=zero,
            entropy=zero,
            regularization=zero,
        )

    @property
    def _rewards(self) -> RewardStatistics:
        """Expose only the filled portion of the reward buffers."""
        n = self._reward_count

        return RewardStatistics(
            rewards=self._reward_buf[:n],  # type: ignore
            lengths=self._length_buf[:n],  # type: ignore
        )

    def record(
        self,
        ep_return: float,
        ep_length: float,
        losses: LossStatistics,
    ) -> None:
        """
        Record one trainer's results into the next available slot.

        Parameters
        ----------
        ep_return : float
            Episodic return
        ep_length : float
            Episode length
        losses : LossStatistics
            Training losses
        """
        i = self._idx

        if ep_return is not None:
            self._reward_buf = self._reward_buf.at[i].set(jnp.float32(ep_return))  # type: ignore
            self._length_buf = self._length_buf.at[i].set(jnp.float32(ep_length))  # type: ignore
            self._reward_count += 1

        # Write all four loss fields in one tree operation
        self._losses = jax.tree.map(
            lambda acc, v: acc.at[i].set(v),
            self._losses,
            losses,
        )
        self._idx += 1

    def summary(self, cat: str = "") -> Dict[str, float]:
        """
        Scalar summary statistics for logging.

        Computes `mean`, `std`, `min`, `max` for both rewards and lengths.
        Returns zeros for all fields if no episodes completed.

        Parameters
        ----------
        cat : str (optional)
            An optional category for the statistics. E.g., `"meta/"` =
            `"meta/avg_reward"`
        """
        return self._rewards.summary(cat)

    def rewards_as_dict(self) -> Dict[str, float]:
        """
        Extracts reward statistics as a dictionary.

        Returns
        -------
        stats : Dict[str, float]
            Reward statistics as a dictionary
        """
        return self._rewards.core_stats()

    def losses_as_dict(self) -> Dict[str, float]:
        """
        Extract loss statistics as a dictionary.

        Returns
        -------
        stats : Dict[str, float]
            Loss statistics as a dictionary
        """
        mean_losses: LossStatistics = jax.tree.map(jnp.mean, self._losses)
        return mean_losses.to_dict()
