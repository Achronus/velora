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
from flax import nnx
from flax.typing import Initializer

from velora.lnn.constants import DEFAULT_BIAS_INIT, DEFAULT_HIDDEN_INIT


class SparseLinear(nnx.Module):
    """
    A linear layer with sparsely weighted connections.

    Equation:
    $$
    y = x * (w * m) + b
    $$

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    mask : jax.Array
        Sparsity mask (m) tensor of shape
        `(out_features, in_features)`
    rngs : flax.nnx.Rngs (optional)
        Random number generator key.
        Must have a `params=[value]` attribute
    hidden_init : flax.nnx.nn.initializers (optional)
        Initializer function for the weight matrix.
        Default is `lecun_uniform()`
    bias_init : flax.nnx.nn.initializers (optional)
        Initializer function for the bias.
        Default is `zeros_init()`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: jax.Array,
        *,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
        hidden_init: Initializer = DEFAULT_HIDDEN_INIT,
        bias_init: Initializer = DEFAULT_BIAS_INIT,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.mask = mask
        self.kernel_init = hidden_init
        self.bias_init = bias_init

        weight_key = rngs.params()
        weights = hidden_init(weight_key, (in_features, out_features))
        self.weights = nnx.Param(weights * self.mask)

        bias_key = rngs.params()
        self.bias = nnx.Param(bias_init(bias_key, (out_features,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Applies a linear transformation to the inputs along the last dimension.

        Parameters
        ----------
        x : jax.Array
            The array to transform with shape `(..., in_features)`

        Returns
        -------
        y_pred : jax.Array
            The layer prediction with sparsity applied. Has shape `(..., out_features)`
        """
        return x @ (self.weights * self.mask) + self.bias
