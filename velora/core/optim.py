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


def scale_by_adam_no_denom(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """
    Adam grad rescaling; but denominator does not receive meta-gradients.

    References:
        [Oh et al., 2025 (GitHub)](https://github.com/google-deepmind/disco_rl/)
        [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

    Parameters:
        b1 (float): decay rate for the exponentially weighted
            average of grads. Default is `0.9`
        b2 (float): decay rate for the exponentially weighted
            average of squared grads. Default is `0.999`
        eps (float): term added to the denominator to improve
            numerical stability. Default is `1e-8`

    Returns:
        transform (optax.GradientTransformation): returns a new object
        with an `init_fn` and `update_fn`
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
