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

import jax
import jax.numpy as jnp


class Softmax:
    """
    Categorical distribution parameterized by unnormalized log-probabilities
    (logits).

    Drop in replacement for `distrax.Softmax`.

    Parameters
    ----------
    logits : jax.Array
        Unnormalized log-probabilities. Shape: `(..., A)` where `A` is
        the number of discrete actions.
    """

    def __init__(self, logits: jax.Array) -> None:
        self._logits = logits
        self._log_probs = jax.nn.log_softmax(logits, axis=-1)

    @property
    def logits(self) -> jax.Array:
        """Raw logits."""
        return self._logits

    @property
    def probs(self) -> jax.Array:
        """Normalized probabilities."""
        return jax.nn.softmax(self._logits, axis=-1)

    def log_prob(self, value: jax.Array) -> jax.Array:
        """
        Log-probability of a discrete action under this distribution.

        Parameters
        ----------
        value : jax.Array
            Integer action indices. Shape: `(...)`

        Returns
        -------
        log_p : jax.Array
            Log-probabilities for the given actions. Shape: `(...)`
        """
        # One-hot encode and dot with log_probs (handles arbitrary batch dims)
        one_hot = jax.nn.one_hot(value, self._log_probs.shape[-1])
        return jnp.sum(self._log_probs * one_hot, axis=-1)

    def entropy(self) -> jax.Array:
        """
        Shannon entropy of the distribution.

        Returns
        -------
        h : jax.Array
            Entropy values. Shape: `(...)` (all dims except the last).
        """
        probs = jax.nn.softmax(self._logits, axis=-1)
        return -jnp.sum(probs * self._log_probs, axis=-1)
