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
import rlax


def compute_kl_loss(
    target: chex.Array,
    preds: chex.Array,
) -> chex.Array:
    """
    Compute KL divergence loss between target and predicted logits.

    Uses softmax to convert logits to probabilities internally.

    Parameters
    ----------
    target : chex.Array
        Target logits from the meta-network
    preds : chex.Array
        Predicted logits from the policy agent

    Returns
    -------
    loss : chex.Array
        Per-element KL divergence loss
    """
    return rlax.categorical_kl_divergence(target, preds)


def compute_aux_policy_loss(
    aux_pi: chex.Array,
    next_pi: chex.Array,
    discounts: chex.Array,
) -> chex.Array:
    """
    Compute auxiliary 1-step policy prediction loss.

    The auxiliary policy `p(s, a)` predicts what the policy will be at the
    next timestep, masked by episode boundaries.

    Parameters
    ----------
    aux_pi : chex.Array
        Auxiliary policy predictions for action taken `(T, B, A)`
    next_pi : chex.Array
        Actual policy at next timestep `(T, B, A)`
    discounts : chex.Array
        Episode continuation signals `(T, B)`

    Returns
    -------
    loss : chex.Array
        Per-timestep auxiliary policy loss `(T, B)`
    """
    loss = compute_kl_loss(jax.lax.stop_gradient(next_pi), aux_pi)

    # Mask out terminal states
    return loss * discounts
