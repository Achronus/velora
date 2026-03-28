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

import chex
import jax
import jax.numpy as jnp

from velora.compute.softmax import Softmax
from velora.compute.utils import batched_index


def compute_entropy_loss(logits: chex.Array, coef: float = 1e-2) -> chex.Array:
    """
    Compute entropy loss for policy regularization.

    Encourages exploration by penalizing low-entropy (overly deterministic)
    policies. Returns negative entropy so that minimizing the loss
    maximizes entropy.

    Parameters
    ----------
    logits : chex.Array
        Policy logits. Shape: `(B, T, A)` or `(B, A)`

        - batch_size (`B`) - the number of samples per timestep
        - seq_length (`T`) - the number of timesteps (optional)
        - n_actions (`A`) - the number of discrete actions

    coef : float (optional)
        Entropy coefficient. Higher values promote more exploration.
        Default is `0.01`

    Returns
    -------
    loss : chex.Array
        Scalar negative mean entropy loss
    """
    return -coef * jnp.mean(Softmax(logits).entropy())


def compute_policy_gradient_loss(
    logits: chex.Array,
    actions: chex.Array,
    advantages: chex.Array,
) -> chex.Array:
    """
    Compute differentiable policy gradient loss (REINFORCE-style).

    Calculates `-log π(a|s) * A(s, a)` for each timestep. Advantages are
    stopped from gradient flow as they should not influence the meta-network
    through this path.

    Parameters
    ----------
    logits : chex.Array
        Policy logits. Shape: `(B, T, A)`
    actions : chex.Array
        Actions taken. Shape: `(B, T, 1)`
    advantages : chex.Array
        Advantage estimates. Shape: `(T, B)`

    Returns
    -------
    loss : chex.Array
        Per-timestep policy gradient loss. Shape: `(B, T)`
    """
    logits = logits[:, :-1]  # (B, T-1), # type: ignore
    actions = actions[:, :-1]  # (B, T-1, 1), # type: ignore

    advantages = jnp.transpose(advantages)  # (B, T)
    actions = jnp.squeeze(actions, axis=-1)  # (B, T)

    log_pi = jax.nn.log_softmax(logits)
    log_pi_a = batched_index(log_pi, actions)
    return -log_pi_a * jax.lax.stop_gradient(advantages)  # type: ignore
