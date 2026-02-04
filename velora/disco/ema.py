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

import functools
from typing import Self, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from velora.disco.settings import EMASettings


@struct.dataclass
class EMAState:
    """
    Exponential Moving Average (EMA) state.

    Parameters
    ----------
    moment1 : jax.ArrayTree
        The first set of moments (mean estimate)
    moment2 : jax.ArrayTree
        The second set of moments (variance estimate)
    decay_product : jax.Array
        Accumulated decay for bias correction
    """

    moment1: chex.ArrayTree
    moment2: chex.ArrayTree
    decay_product: chex.Array

    @classmethod
    def create(cls) -> Self:
        """
        Create initial EMA state.

        Returns
        -------
        state : EMAState
            Initial EMA state
        """
        return cls(
            moment1=jnp.zeros(()),
            moment2=jnp.zeros(()),
            decay_product=jnp.ones(()),
        )


class MovingAverage:
    """
    Implements the Exponential Moving Average (EMA).

    Parameters
    ----------
    config : EMASettings
        Configuration settings for the EMA
    """

    def __init__(self, config: EMASettings) -> None:
        self._decay = config.decay
        self._eps = config.eps
        self._root_eps = config.root_eps

    def update(
        self,
        x: chex.ArrayTree,
        state: EMAState,
        pmean_axis_name: str | None = None,
    ) -> EMAState:
        """
        Updates the EMA state and returns it.

        Parameters
        ----------
        x : jax.ArrayTree
            Data to use to update state
        state : EMAState
            Current EMA state
        pmean_axis_name : str (optional)
            The optional axis name for parallel mean computation across multiple devices. Default is `None`

        Returns
        -------
        new_state : EMAState
            The updated state
        """
        squared_tree = jax.tree.map(jnp.square, x)

        def _update(
            moment: chex.Array,
            x: chex.Array,
            pmean_axis_name: str | None = None,
        ) -> chex.Array:
            """
            Computes the mean across all learner devices involved in the `pmap`.

            Parameters
            ----------
            moment : jax.Array
                EMA moment array
            x : jax.Array
                Data array
            pmean_axis_name : str (optional)
                The optional axis name for parallel mean computation across multiple devices. Default is `None`
            """
            mean = jnp.mean(x)

            if pmean_axis_name is not None:
                mean = jax.lax.pmean(mean, axis_name=pmean_axis_name)

            return self._decay * moment + (1.0 - self._decay) * mean

        update_fn = functools.partial(_update, pmean_axis_name=pmean_axis_name)
        moment1 = jax.tree.map(update_fn, state.moment1, x)
        moment2 = jax.tree.map(update_fn, state.moment2, squared_tree)

        return EMAState(
            moment1=moment1,
            moment2=moment2,
            decay_product=state.decay_product * self._decay,
        )

    def _compute_moments(
        self, state: EMAState
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """
        Computes moments `(mean, variance)`, applying 0-debiasing like in the Adam optimizer.

        Accounts for the initial moments set to 0 and estimates the
        zero-centered debiased variance with negative values clipped to
        safeguard against numerical errors.

        Parameters
        ----------
        state : EMAState
            Current EMA state

        Returns
        -------
        m1 : jax.ArrayTree
            The computed mean (moment 1)
        m2 : jax.ArrayTree
            The computed variance (moment 2)
        """
        debias = 1.0 / (1 - state.decay_product)

        mean = jax.tree.map(lambda m1: m1 * debias, state.moment1)

        variance = jax.tree.map(
            lambda m2, m: jnp.maximum(0.0, m2 * debias - jnp.square(m)),
            state.moment2,
            mean,
        )

        return mean, variance

    def normalize(
        self,
        x: chex.ArrayTree,
        state: EMAState,
        subtract_mean: bool = True,
    ) -> chex.Array:
        """
        Normalizes `x` by dividing by the second moment and subtracting its mean.

        Uses two epsilons for numerical stability when backpropagation through
        the normalization (like in `optax.scale_by_adam`).

        Parameters
        ----------
        x : jax.ArrayTree
            Data to normalize
        state : EMAState
            Current EMA state
        subtract_mean : bool (optional)
            A flag for mean subtraction. Default is `True`

        Returns
        -------
        x_normalized : jax.Array
            Normalized data
        """

        def _normalize(mean, var, val) -> chex.ArrayTree:
            calc1 = jnp.sqrt(var + self._root_eps)

            if subtract_mean:
                return (val - mean) / (calc1 + self._eps)

            return val / (calc1 + self._eps)

        mean, variance = self._compute_moments(state)
        return jax.tree.map(_normalize, mean, variance, x)

    def update_and_normalize(
        self,
        x: chex.ArrayTree,
        state: EMAState,
        subtract_mean: bool = True,
        pmean_axis_name: str | None = None,
    ) -> Tuple[chex.Array, EMAState]:
        """
        Updates EMA state and then normalizes `x` using it.

        Parameters
        ----------
        x : jax.ArrayTree
            Data to use to update EMA state and normalize
        state : EMAState
            Current EMA state
        subtract_mean : bool (optional)
            A flag for mean subtraction. Default is `True`
        pmean_axis_name : str (optional)
            The optional axis name for parallel mean computation across multiple devices. Default is `None`

        Returns
        -------
        norm : jax.Array
            Normalized x
        new_state : EMAState
            Updated state
        """
        new_state = self.update(x, state, pmean_axis_name)
        return self.normalize(x, new_state, subtract_mean), new_state
