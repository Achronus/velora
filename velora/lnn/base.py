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

from dataclasses import fields
from typing import Optional, Tuple, final

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from velora.base.spec import HeadSpec, LayerSpec
from velora.lnn.cell import NCPLiquidCell
from velora.lnn.constants import DEFAULT_HIDDEN_INIT
from velora.lnn.spec import NCPWiringSpec
from velora.utils.nn import active_parameters, total_parameters
from velora.utils.transforms import to_batch_first, to_time_first


class BaseCfC(nnx.Module):
    """
    Base class for CfC-LNN NCP-based models.

    Designed to be used as a backbone for LNN model creation with method overrides.

    Required method overrides -
        - `_build_wiring()` - set the network wiring with `N` output heads using the `NCPWiringBuilder` class

    Optional method overrides -
        - `_preprocess` - performs preprocessing techniques on network inputs
        - `_postprocess` - performs postprocessing techniques on network predictions

    Internal methods (do not override) -
        - `_init_motor()` - Dynamically creates heads from wiring
        - `_scan()` - Core forward pass logic

    Layers -
        1. Inter (input) - a `NCPLiquidCell` layer
        2. Command (hidden) - a `NCPLiquidCell` layer
        3. Motor (output) - one or more `NCPLiquidCell` layers controlled through subclasses

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
        Number of decision nodes (inter + command nodes)
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

    Raises
    ------
    not_implemented : NotImplementedError
        When `_build_wiring()` is not implemented
    """

    _head_cells: nnx.List[NCPLiquidCell]

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        *,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
        init_type: Initializer = DEFAULT_HIDDEN_INIT,
    ) -> None:
        self.in_features = in_features
        self.n_neurons = n_neurons
        self.sparsity = sparsity
        self.init_type = init_type
        self.key = key

        self.seed = jax.random.key_data(key)[-1].item()
        self.rngs = nnx.Rngs(params=self.key)

        self.wiring: NCPWiringSpec = nnx.data(self._build_wiring())

        self.hidden_size = self.wiring.hidden_size
        self.hidden_split_indices = nnx.data(self.wiring.h_split_indices())

        # Inter layer: sensory -> inter
        self.inter = NCPLiquidCell(
            self.in_features,
            self.wiring.inter.n_hidden,
            self.wiring.inter.mask,
            rngs=self.rngs,
            init_type=self.init_type,
        )

        # Command layer: inter -> command
        self.command = NCPLiquidCell(
            self.wiring.inter.n_hidden,
            self.wiring.command.n_hidden,
            self.wiring.command.mask,
            rngs=self.rngs,
            init_type=self.init_type,
        )

        # Motor heads: command -> motors (outputs)
        self.motor: HeadSpec = nnx.data(self.wiring.motor)
        self._init_motor()

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

    @final
    def _init_motor(self) -> None:
        """
        Dynamically initialize motor layer heads based on wiring.

        Stores `NCPLiquidCell` heads in `self._head_cells` with the naming convention `[field_name]_head` from wiring `HeadSpec`.
        """
        head_cells = []

        for field in fields(self.motor):
            name = field.name
            layer_spec: LayerSpec = getattr(self.motor, name)

            head = NCPLiquidCell(
                self.wiring.command.n_hidden,
                layer_spec.n_hidden,
                layer_spec.mask,
                rngs=self.rngs,
                init_type=self.init_type,
            )

            # Set heads as attributes
            setattr(self, f"{name}_head", head)
            head_cells.append(head)

        self._head_cells = nnx.List(head_cells)

    @final
    def _scan(
        self,
        x: chex.Array,
        h_state: chex.Array,
        timespans: chex.Array,
        reverse: bool = False,
    ) -> Tuple[chex.Array, Tuple[chex.Array, ...]]:
        """
        Run forward scan through network.

        Should be used in the `__call__` method after `_preprocess`.

        Parameters
        ----------
        x : jax.Array
            Preprocessed input.
        h_state : jax.Array
            Preprocessed hidden state.
        timespans : jax.Array
            Preprocessed timespans.
        reverse : bool (optional)
            A flag to reverse input values (`x`, `timespans`) for bootstrapping. Default is `False`

        Returns
        -------
        h_state : chex.Array
            Updated hidden state.
        preds : Tuple[chex.Array, ...]
            Network predictions. Includes:
            - `embeddding` - output of the command layer
            - `n_head_outputs` - `n` additional outputs, one per motor head
        """
        T = jnp.shape(x)[1]

        def _step(
            h: Tuple[chex.Array, ...],
            inputs: Tuple[chex.Array, chex.Array],
        ) -> Tuple[Tuple[chex.Array, ...], Tuple[chex.Array, ...]]:
            """Single step function."""
            x_t, ts_t = inputs  # x_t -> (B, F), ts_t -> scalar
            h_inter, h_command, *h_heads = h

            # Forward through core liquid layers
            x_t, new_h_inter = self.inter(x_t, h_inter, ts_t)
            embed_t, new_h_command = self.command(x_t, h_command, ts_t)

            # Forward through each head
            head_results = tuple(
                head(embed_t, h_head, ts_t)
                for head, h_head in zip(self._head_cells, h_heads)
            )
            outputs, new_h_heads = zip(*head_results)

            new_h = (new_h_inter, new_h_command, *new_h_heads)
            return new_h, (embed_t, *outputs)  # type: ignore

        # Transpose for scanning over time: (B, T, F) -> (T, B, F)
        x_t = to_time_first(x)

        # Enable bootstrapping
        if reverse:
            x_t = jnp.flipud(x_t)
            timespans = jnp.flipud(timespans)

        h_split = tuple(jnp.split(h_state, self.hidden_split_indices, axis=1))
        new_h, preds = jax.lax.scan(_step, h_split, (x_t, timespans), length=T)

        # Reverse preds back to forward order
        if reverse:
            preds = jax.tree.map(lambda x: jnp.flipud(x), preds)

        h_state = jnp.concatenate(new_h, axis=1)  # (B, H)
        return h_state, preds

    def _build_wiring(self) -> NCPWiringSpec:
        """
        Build the NCP wiring specification.

        Returns
        -------
        wiring : NCPWiringSpec
            NCP wiring specification

        Examples
        --------
        Creating wiring with 1 output head:
        ```python
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
        ```
        """
        raise NotImplementedError()

    def _preprocess(
        self,
        x: chex.Array,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Overridable method. Preprocesses `__call__` method inputs.

        Includes -
        - `x` dimension expansion from `(B, F)` -> `(B, T, F)` (if needed)
        - `h_state` initialized to `(B, H)` when set to `None`
        - `timespans` initialized to `(T,)` of `1s` when set to `None`

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
        """
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=1)

        B, T, F = jnp.shape(x)

        if h_state is None:
            h_state = jnp.zeros((B, self.hidden_size))

        timespans = jnp.ones(T) if timespans is None else timespans

        return x, h_state, timespans

    def _postprocess(self, preds: Tuple[chex.Array, ...]) -> Tuple[chex.Array, ...]:
        """
        Overridable method. Postprocess network predictions by applying batch-first transformations to all predictions.

        Should be used in the `__call__` method after `_scan`.

        Parameters
        ----------
        preds : Tuple[chex.Array, ...]
            Raw predictions from scan `(T, B, F)`

        Returns
        -------
        outputs : Tuple[chex.Array, ...]
            Transformed predictions `(B, T, F)`
        """
        return jax.tree.map(to_batch_first, preds)
