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
import jax.numpy as jnp
import rlax
from flax import nnx, struct


@struct.dataclass(frozen=True)
class CategoricalBins:
    """
    Categorical bin values for distributional Q-values.

    Handles conversion between scalar values and categorical distributions
    using 2-hot encoding for differentiable, soft bin assignments via the
    `rlax` library.

    Parameters
    ----------
    support : jax.Array
        Array of bin center values `(num_bins,)`
    min_value : float
        Minimum representable value
    max_value : float
        Maximum representable value
    """

    support: chex.Array
    min_value: float
    max_value: float

    @property
    def num_bins(self) -> int:
        """Number of categorical bins."""
        return jnp.shape(self.support)[0]

    @property
    def resolution(self) -> float:
        """Width of each bin."""
        return (self.max_value - self.min_value) / (self.num_bins - 1)

    def to_probs(self, scalar: chex.Array) -> chex.Array:
        """
        Convert scalar values to categorical distribution using 2-hot encoding.

        Parameters
        ----------
        scalar : jax.Array
            Scalar values to convert

        Returns
        -------
        probs : jax.Array
            Categorical distribution over bins
        """
        return rlax.transform_to_2hot(
            scalar,
            self.min_value,
            self.max_value,
            self.num_bins,
        )

    def to_scalar(self, probs: chex.Array) -> chex.Array:
        """
        Convert categorical distribution to scalar (expected value).

        Parameters
        ----------
        probs : jax.Array
            Probability distribution over bins

        Returns
        -------
        scalar : jax.Array
            Expected scalar values
        """
        return rlax.transform_from_2hot(
            probs,
            self.min_value,
            self.max_value,
            self.num_bins,
        )

    def to_scalar_from_logits(self, logits: chex.Array) -> chex.Array:
        """
        Convert logits to scalar via softmax then expected value.

        Parameters
        ----------
        logits : jax.Array
            Unnormalized log probabilities

        Returns
        -------
        scalar : jax.Array
            Expected scalar values
        """
        return self.to_scalar(nnx.softmax(logits, axis=-1))
