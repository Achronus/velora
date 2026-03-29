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

from typing import Dict, Tuple

import chex
import jax.numpy as jnp
from flax import nnx

from velora.disco.outputs import ActionDecoderOutput
from velora.utils.nn import active_parameters, total_parameters


class ActionDecoder(nnx.Module):
    """
    Shared-weight decoder that produces per-action outputs from fixed-size
    hidden representations using the DiscoRL action-invariant pattern.

    Concatenates all hidden sources (from OCM and ACM) into a single vector,
    broadcasts per-action with identity embeddings, projects through a shared
    `nnx.Linear`, and slices the output into named heads.

    Weight shapes are fixed at `(in_dim + max_actions, out_dim)` regardless
    of an environment's actual action count (`vmap` compatible).

    Parameters
    ----------
    in_dim : int
        Total dimension of the concatenated hidden inputs
    max_actions : int
        Maximum number of discrete actions across all environments in the
        training set
    pi_dim : int
        Output dimension for the policy head. Each action gets a single
        logit, so this is typically `1`
    z_dim : int
        Output dimension for the action-conditioned prediction head.
        Matches `prediction_size`
    aux_pi_dim : int
        Output dimension for the auxiliary policy prediction head.
        Set to `max_actions` so the auxiliary policy predicts a full
        distribution over the padded action space
    q_dim : int
        Output dimension for the action-value prediction head.
        Matches `num_bins` for distributional Q-values
    key : chex.PRNGKey
        Random number generator key
    """

    def __init__(
        self,
        in_dim: int,
        max_actions: int,
        *,
        pi_dim: int,
        z_dim: int,
        aux_pi_dim: int,
        q_dim: int,
        key: chex.PRNGKey,
    ) -> None:
        self.rngs = nnx.Rngs(key)
        self.max_actions = max_actions

        head_dims = {"pi": pi_dim, "z": z_dim, "aux_pi": aux_pi_dim, "q": q_dim}
        out_dim = sum(head_dims.values())

        self.proj = nnx.Linear(
            in_dim + max_actions,
            out_dim,
            rngs=self.rngs,
        )

        # Pre-compute slice boundaries for each head
        self._slices = self._build_slices(head_dims)

        self._action_ids = jnp.eye(max_actions)

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns
        -------
        count : int
            The total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns
        -------
        count : int
            The active parameter count.
        """
        return self._active_params

    @staticmethod
    def _build_slices(head_dims: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
        """
        Compute start/end indices for each named head.

        Parameters
        ----------
        head_dims : Dict[str, int]
            Head name to output dimension mapping

        Returns
        -------
        slices : Dict[str, Tuple[int, int]]
            Head name to `(start, end)` index pair
        """
        slices = {}
        offset = 0
        for name, dim in head_dims.items():
            slices[name] = (offset, offset + dim)
            offset += dim

        return slices

    def _fuse_hidden(self, *sources: chex.Array) -> chex.Array:
        """
        Concatenate multiple hidden representations along the feature axis.

        Parameters
        ----------
        *sources : chex.Array
            Hidden representations to fuse. All must share leading
            dimensions `(B, T, ...)` or `(B, ...)`.

        Returns
        -------
        fused : chex.Array
            Concatenated hidden vector. Shape: `(B, T, D)` or `(B, D)`
            where `D` is the sum of all source feature dimensions
        """
        return jnp.concatenate(sources, axis=-1)

    def _broadcast_with_action_ids(self, hidden: chex.Array) -> chex.Array:
        """
        Broadcast hidden state per-action and concatenate action identity
        embeddings.

        Each action receives the same hidden state plus a unique one-hot
        identity vector (a row from `jnp.eye(max_actions)`), giving
        the shared projection layer enough information to produce
        action-specific outputs.

        Parameters
        ----------
        hidden : chex.Array
            Fused hidden representation. Shape: `(B, T, H)` or `(B, H)`

        Returns
        -------
        combined : chex.Array
            Per-action input. Shape: `(B, T, n, H + max_actions)`
            or `(B, n, H + max_actions)`
        """
        n = self.max_actions
        action_ids = self._action_ids

        if hidden.ndim == 3:
            B, T, H = jnp.shape(hidden)
            h_expanded = jnp.broadcast_to(hidden[:, :, None, :], (B, T, n, H))
            a_expanded = jnp.broadcast_to(action_ids[None, None, :, :], (B, T, n, n))
        else:
            B, H = jnp.shape(hidden)
            h_expanded = jnp.broadcast_to(hidden[:, None, :], (B, n, H))  # type: ignore
            a_expanded = jnp.broadcast_to(action_ids[None, :, :], (B, n, n))

        return jnp.concatenate([h_expanded, a_expanded], axis=-1)

    def __call__(self, *sources: chex.Array) -> ActionDecoderOutput:
        """
        Decode fixed-size hidden representations to per-action outputs.

        Parameters
        ----------
        *sources : chex.Array
            Hidden representations to fuse and decode. Typically
            `(ocm_preds.pi, acm_preds.z, acm_preds.aux_pi, acm_preds.q)`.
            All must share leading dimensions `(B, T)` or `(B,)`

        Returns
        -------
        output : ActionDecoderOutput
            Per-action outputs at `max_actions` dimension
        """
        hidden = self._fuse_hidden(*sources)
        combined = self._broadcast_with_action_ids(hidden)
        raw = self.proj(combined)  # type: ignore

        s = self._slices

        return ActionDecoderOutput(
            pi=raw[..., s["pi"][0] : s["pi"][1]],
            z=raw[..., s["z"][0] : s["z"][1]],
            aux_pi=raw[..., s["aux_pi"][0] : s["aux_pi"][1]],
            q=raw[..., s["q"][0] : s["q"][1]],
        )


class GaussianPolicyDecoder(nnx.Module):
    """
    Projects LNN hidden representations to Gaussian policy outputs.

    Uses three layer projections:
        - `mu_proj`: OCM pi_hidden → policy mean `μ(s)`
        - `log_std_proj`: OCM pi_hidden → policy log-std `log σ(s)`
        - `aux_pi_proj`: ACM aux_pi hidden → predicted next-step `(μ', log σ')`

    Applies log-std clamping and action dimension masking internally.

    Parameters
    ----------
    prediction_size : int
        Input dimension (OCM/ACM head output size)
    max_action_dim : int
        Output width for mu/log_std (padded action dimensionality)
    min_log_std : float
        Minimum log standard deviation clamp
    max_log_std : float
        Maximum log standard deviation clamp
    key : chex.PRNGKey
        Random number generator key
    """

    def __init__(
        self,
        prediction_size: int,
        max_action_dim: int,
        *,
        min_log_std: float,
        max_log_std: float,
        key: chex.PRNGKey,
    ) -> None:
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.rngs = nnx.Rngs(key)

        self.mu_proj = nnx.Linear(prediction_size, max_action_dim, rngs=self.rngs)
        self.log_std_proj = nnx.Linear(prediction_size, max_action_dim, rngs=self.rngs)
        self.aux_pi_proj = nnx.Linear(
            prediction_size,
            2 * max_action_dim,
            rngs=self.rngs,
        )

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns
        -------
        count : int
            The total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns
        -------
        count : int
            The active parameter count.
        """
        return self._active_params

    def __call__(
        self,
        pi_hidden: chex.Array,
        aux_pi_hidden: chex.Array,
        *,
        action_dim_mask: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Project hidden representations to policy outputs.

        Parameters
        ----------
        pi_hidden : chex.Array
            OCM policy hidden representation. Shape: `(B, T, H)`
        aux_pi_hidden : chex.Array
            ACM auxiliary policy hidden representation. Shape: `(B, T, H)`
        action_dim_mask : chex.Array
            Boolean mask `(D,)` for valid action dimensions

        Returns
        -------
        mu : chex.Array
            Policy mean `(B, T, D)`, masked and zero-padded
        log_std : chex.Array
            Policy log-std `(B, T, D)`, clamped and masked
        aux_pi : chex.Array
            Predicted next-step Gaussian params `(B, T, 2D)`
        """
        mu = self.mu_proj(pi_hidden)  # type: ignore
        log_std = jnp.clip(
            self.log_std_proj(pi_hidden),  # type: ignore
            self.min_log_std,
            self.max_log_std,
        )

        mu = jnp.where(action_dim_mask, mu, 0.0)
        log_std = jnp.where(action_dim_mask, log_std, self.min_log_std)

        aux_pi = self.aux_pi_proj(aux_pi_hidden)  # type: ignore

        return mu, log_std, aux_pi
