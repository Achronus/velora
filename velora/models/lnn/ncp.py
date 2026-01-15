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

from typing import Optional, Tuple

import chex
from flax.typing import Initializer

from velora.config.spec import NCPWiringSpec, SingleHeadSpec
from velora.constants import DEFAULT_HIDDEN_INIT
from velora.models.lnn.base import BaseNCP
from velora.models.lnn.wiring import NCPWiringBuilder


class LNN(BaseNCP):
    """
    A CfC Liquid Neural Circuit Policy (NCP) Network with three layers.

    Layers -
        1. Inter (input) - a `NCPLiquidCell` layer
        2. Command (hidden) - a `NCPLiquidCell` layer
        3. Motor (output) - a `NCPLiquidCell` layer

    ??? note "Decision nodes"

        `inter` and `command` neurons are automatically calculated using:

        ```python
        command_neurons = max(int(0.4 * n_neurons), 1)
        inter_neurons = n_neurons - command_neurons
        ```

    Combines a Liquid Time-Constant (LTC) cell with Ordinary Neural Circuits (ONCs).

    References -
        - [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898)
        - [Reinforcement Learning with Ordinary Neural Circuits](https://proceedings.mlr.press/v119/hasani20a.html)

    Parameters
    ----------
    in_features : int
        Number of inputs (sensory nodes)
    n_neurons : int
        Number of decision nodes (inter and command nodes)
    out_features : int
        Number of out features (motor nodes)
    key : chex.PRNGKey
        Random number generator key
    sparsity : float (optional)
        Controls the connection sparsity between neurons.
        Default is `0.5`.
        Must be a value between `[0.1, 0.9]`:

        - Where `0.1` neurons are very dense
        - Where `0.9` neurons are very sparse
    init_type : flax.nnx.nn.initializers (optional)
        Initializer function for the weight matrix.
        Default is `lecun_uniform()`
    """

    motor: SingleHeadSpec  # type: ignore

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
        init_type: Initializer = DEFAULT_HIDDEN_INIT,
    ) -> None:
        self.out_features = out_features

        super().__init__(
            in_features,
            n_neurons,
            key=key,
            sparsity=sparsity,
            init_type=init_type,
        )

    def _build_wiring(self) -> NCPWiringSpec:
        return (
            NCPWiringBuilder(
                self.in_features,
                self.n_neurons,
                seed=self.seed,
                sparsity=self.sparsity,
            )
            .add_output_heads(
                SingleHeadSpec,
                out=self.out_features,
            )
            .build()
        )

    def __call__(
        self,
        x: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        x : jax.Array
            An input array of shape: `(B, F)` or `(B, T, F)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories, channels)
            - `features (F)` the features at each timestep

        h_state : jax.Array (optional)
            Initial hidden state of the RNN with shape: `(B, H)`

            - `batch_size (B)` the number of samples per timestep
            - `n_hidden (H)` the total number of hidden neurons

        timespans : jax.Array (optional)
            Time elapsed since previous timestep.
            For fixed intervals set to `None`. For varying timesteps shape
            should be `(T,)`

        Returns
        -------
        out_preds : jax.Array
            The network prediction. Shape `(B, T, F)`
        h_state : jax.Array
            The final hidden state. Shape `(B, H)`
        """
        x, h_state, timespans = self._preprocess(x, h_state, timespans)
        h_state, preds = self._scan(x, h_state, timespans)

        # 2 outputs -> (embeddings, out_preds)
        _, out_preds = self._postprocess(preds)
        return out_preds, h_state
