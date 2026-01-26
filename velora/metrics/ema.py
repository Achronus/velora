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
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EMAState:
    """
    Exponential Moving Average (EMA) state.

    Parameters
    ----------
    moment1 : jax.ArrayTree
        The first set of moments
    moment2 : jax.ArrayTree
        The second set of moments
    decay_product : jax.Array
        The product of all decays from start of accumulation
    """

    moment1: chex.ArrayTree
    moment2: chex.ArrayTree
    decay_product: chex.Array


class MovingAverage:
    """
    Implements the Exponential Moving Average (EMA).

    Parameters
    ----------
    x : jax.ArrayTree (optional)
        An example array tree structure used to update the EMA state. When `None` automatically creates a scalar array. Default is `None`
    decay : float (optional)
        The learning rate (moment) decay. Default is `0.999`
    eps : float (optional)
        Epsilon used for normalization. Default is `1e-6`
    """

    def __init__(
        self,
        x: chex.ArrayTree | None = None,
        *,
        decay: float = 0.999,
        eps: float = 1e-6,
    ) -> None:
        self._x = jnp.zeros(()) if x is None else x
        self._decay = decay
        self._eps = eps

        self._state = self._init_state()

    @property
    def state(self) -> EMAState:
        """
        Current EMA state.
        """
        return self._state

    def _init_state(self) -> EMAState:
        """
        Initializes the EMA state.

        Returns
        -------
        state : EMAState
            A new EMA state object
        """
        return EMAState(
            moment1=jnp.zeros(()),
            moment2=jnp.zeros(()),
            decay_product=jnp.ones([]),
        )

    def update(
        self,
        x: chex.ArrayTree,
        pmean_axis_name: str | None = None,
    ) -> None:
        """
        Updates the EMA state.

        Parameters
        ----------
        x : jax.ArrayTree
            Data to use to update state
        pmean_axis_name : str (optional)
            The optional axis name for parallel mean computation across multiple devices. Default is `None`
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
        moment1 = jax.tree.map(update_fn, self._state.moment1, x)
        moment2 = jax.tree.map(update_fn, self._state.moment2, squared_tree)

        self._state = EMAState(
            moment1=moment1,
            moment2=moment2,
            decay_product=self._state.decay_product * self._decay,
        )

    def _compute_moments(self) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """
        Computes moments `(mean, variance)`, applying 0-debiasing like in the Adam optimizer.

        Accounts for the initial moments set to 0 and estimates the
        zero-centered debiased variance with negative values clipped to
        safeguard against numerical errors.

        Returns
        -------
        m1 : jax.ArrayTree
            The computed mean (moment 1)
        m2 : jax.ArrayTree
            The computed variance (moment 2)
        """
        debias = 1.0 / (1 - self._state.decay_product)

        mean = jax.tree.map(lambda m1: m1 * debias, self._state.moment1)

        variance = jax.tree.map(
            lambda m2, m: jnp.maximum(0.0, m2 * debias - jnp.square(m)),
            self._state.moment2,
            mean,
        )

        return mean, variance

    def normalize(
        self,
        x: chex.ArrayTree,
        subtract_mean: bool = True,
        root_eps: float = 1e-12,
    ) -> chex.Array:
        """
        Normalizes `x` by dividing by the second moment and subtracting its mean.

        Uses two epsilons for numerical stability when backpropagation through
        the normalization (like in `optax.scale_by_adam`).

        Parameters
        ----------
        x : jax.ArrayTree
            Data to normalize
        subtract_mean : bool, optional
            A flag for mean subtraction. Default is `True`
        root_eps : float, optional
            Primary epsilon value. Default is `1e-12`

        Returns
        -------
        x_normalized : jax.Array
            Normalized data
        """

        def _normalize(mean, var, val) -> chex.ArrayTree:
            calc1 = jnp.sqrt(var + root_eps)

            if subtract_mean:
                return (val - mean) / (calc1 + self._eps)

            return val / (calc1 + self._eps)

        mean, variance = self._compute_moments()
        return jax.tree.map(_normalize, mean, variance, x)

    def update_and_normalize(
        self,
        x: chex.ArrayTree,
        subtract_mean: bool = True,
        root_eps: float = 1e-12,
        pmean_axis_name: str | None = None,
    ) -> chex.Array:
        """
        Updates EMA state and then normalizes `x` using it.

        Parameters
        ----------
        x : jax.ArrayTree
            Data to use to update EMA state and normalize
        subtract_mean : bool, optional
            A flag for mean subtraction. Default is `True`
        root_eps : float, optional
            Primary epsilon value. Default is `1e-12`
        pmean_axis_name : str (optional)
            The optional axis name for parallel mean computation across multiple devices. Default is `None`

        Returns
        -------
        norm : jax.Array
            Normalized x
        """
        self.update(x, pmean_axis_name)
        return self.normalize(x, subtract_mean, root_eps)

    def reset(self) -> None:
        """
        Resets the EMA state to initial values.
        """
        self._state = self._init_state()

    def load_state(self, new_state: EMAState) -> None:
        """
        Load a state from a checkpoint.

        Parameters
        ----------
        new_state : EMAState
            A new EMA state
        """
        self._state = new_state
