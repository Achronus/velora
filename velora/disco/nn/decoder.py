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

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import nnx

from velora.utils.nn import active_parameters, total_parameters


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
        pi_hidden: jax.Array,
        aux_pi_hidden: jax.Array,
        *,
        action_dim_mask: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Project hidden representations to policy outputs.

        Parameters
        ----------
        pi_hidden : jax.Array
            OCM policy hidden representation. Shape: `(B, T, H)`
        aux_pi_hidden : jax.Array
            ACM auxiliary policy hidden representation. Shape: `(B, T, H)`
        action_dim_mask : jax.Array
            Boolean mask `(D,)` for valid action dimensions

        Returns
        -------
        mu : jax.Array
            Policy mean `(B, T, D)`, masked and zero-padded
        log_std : jax.Array
            Policy log-std `(B, T, D)`, clamped and masked
        aux_pi : jax.Array
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
