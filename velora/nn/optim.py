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
import optax
from flax import struct


def scale_by_adam_no_denom(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """
    Adam grad rescaling; but denominator does not receive meta-gradients.

    References -
        - [Oh et al., 2025 (GitHub)](https://github.com/google-deepmind/disco_rl/)
        - [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

    Parameters
    ----------
    b1 : float (optional)
        Decay rate for the exponentially weighted average of grads.
        Default is `0.9`
    b2 : float (optional)
        Decay rate for the exponentially weighted average of squared grads.
        Default is `0.999`
    eps : float (optional)
        Term added to the denominator to improve numerical stability.
        Default is `1e-8`

    Returns
    -------
    transform : optax.GradientTransformation
        Returns a new object with an `init_fn` and `update_fn`
    """

    def init_fn(params):
        mu = jax.tree.map(jnp.zeros_like, params)  # First moment.
        nu = jax.tree.map(jnp.zeros_like, params)  # Second moment.
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = optax.update_moment(updates, state.mu, b1, 1)
        nu = optax.update_moment(updates, state.nu, b2, 2)
        count_inc = optax.safe_int32_increment(state.count)
        mu_hat = optax.bias_correction(mu, b1, count_inc)
        nu_hat = optax.bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: m / (jnp.sqrt(v) + eps),
            mu_hat,
            jax.lax.stop_gradient(nu_hat),  # NOTE: stop_gradient on nu_hat here
        )
        return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)  # type: ignore

    return optax.GradientTransformation(init_fn, update_fn)


@struct.dataclass
class ScaleByAdanState:
    """
    State for the Adan algorithm.

    Parameters
    ----------
    count : jax.Array
        Number of update steps taken.
    mu : optax.Updates
        Exponentially weighted average of gradients (first moment).
    nu : optax.Updates
        Exponentially weighted average of gradient differences.
    n : optax.Updates
        Exponentially weighted average of squared NME gradients (second moment).
    prev_grad : optax.Updates
        Gradient from the previous step, used to compute gradient differences.
    """

    count: jax.Array
    mu: optax.Updates
    nu: optax.Updates
    n: optax.Updates
    prev_grad: optax.Updates


def scale_by_adan_no_denom(
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """
    Adan grad rescaling; denominator does not receive meta-gradients.

    References:
        - [Xie et al., 2024](https://arxiv.org/abs/2208.06677)

    Parameters
    ----------
    b1 : float (optional)
        Decay rate for first moment. Default is `0.98`.
    b2 : float (optional)
        Decay rate for gradient difference moment. Default is `0.92`.
    b3 : float (optional)
        Decay rate for second moment. Default is `0.99`.
    eps : float (optional)
        Numerical stability constant. Default is `1e-8`.

    Returns
    -------
    transform : optax.GradientTransformation
    """

    def init_fn(params: optax.Params) -> ScaleByAdanState:
        return ScaleByAdanState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree.map(jnp.zeros_like, params),
            nu=jax.tree.map(jnp.zeros_like, params),
            n=jax.tree.map(jnp.zeros_like, params),
            prev_grad=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(
        updates: optax.Updates,
        state: ScaleByAdanState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, ScaleByAdanState]:
        del params
        count_inc = optax.safe_int32_increment(state.count)

        # Gradient difference: (g_k - g_{k-1})
        grad_diff = jax.tree.map(lambda g, gp: g - gp, updates, state.prev_grad)

        # First moment: m_k
        mu = optax.update_moment(updates, state.mu, b1, 1)

        # Gradient difference moment: v_k
        nu = optax.update_moment(grad_diff, state.nu, b2, 1)

        # NME gradient: g'_k = g_k + (1 - b2) * (g_k - g_{k-1})
        nme_grad = jax.tree.map(lambda g, gd: g + (1.0 - b2) * gd, updates, grad_diff)

        # Second moment: n_k using NME gradient
        n = optax.update_moment(nme_grad, state.n, b3, 2)

        # Bias correction
        mu_hat = optax.bias_correction(mu, b1, count_inc)
        nu_hat = optax.bias_correction(nu, b2, count_inc)
        n_hat = optax.bias_correction(n, b3, count_inc)

        # Update: (m + (1-b2)*v) / (sqrt(n) + eps)
        new_updates = jax.tree.map(
            lambda m, v, n_val: (m + (1.0 - b2) * v) / (jnp.sqrt(n_val) + eps),
            mu_hat,
            nu_hat,
            jax.lax.stop_gradient(n_hat),
        )

        return new_updates, ScaleByAdanState(
            count=count_inc,  # type: ignore
            mu=mu,
            nu=nu,
            n=n,
            prev_grad=updates,
        )

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore
