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
from typing import Dict, Self, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
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
        Policy parameters
    value_outs : ValueOutputs
        Value function outputs
    """

    pg_loss: chex.Array
    entropy_loss: chex.Array
    reg_loss: chex.Array
    disco_h: chex.Array
    meta_h: chex.Array
    p_params: optax.Params
    value_outs: ValueOutputs


@struct.dataclass(frozen=True)
class LossStatistics:
    """
    Loss statistics for meta-training.

    Parameters
    ----------
    meta : float
        Total meta loss
    policy_gradient : float
        Policy gradient loss
    entropy : float
        Entropy loss
    regularization : float
        Regularization loss
    """

    meta: float
    policy_gradient: float
    entropy: float
    regularization: float

    @classmethod
    def from_losses(cls, losses: Sequence[Self]) -> Self:
        """
        Create averaged statistics from a sequence of losses.

        Parameters
        ----------
        losses : Sequence[LossStatistics]
            Sequence of loss statistics

        Returns
        -------
        stats : LossStatistics
            Averaged loss statistics
        """
        n = len(losses)

        return cls(
            meta=sum(loss.meta for loss in losses) / n,
            policy_gradient=sum(loss.policy_gradient for loss in losses) / n,
            entropy=sum(loss.entropy for loss in losses) / n,
            regularization=sum(loss.regularization for loss in losses) / n,
        )

    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary.

        Returns
        -------
        stats : Dict[str, float]
            Loss statistics as a dictionary
        """
        return vars(self)
