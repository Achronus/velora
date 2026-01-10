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


def to_time_first(x: chex.Array) -> chex.Array:
    """
    Transpose from batch-first to time-first format for scan operations.

    Parameters
    ----------
    x : jax.Array
        Input tensor in batch-first format `(B, T, ...)`

    Returns
    -------
    x : jax.Array
        Output tensor in time-first format `(T, B, ...)`
    """
    # (B, T, ...) -> (T, B, ...)
    return jnp.swapaxes(x, 0, 1)


def to_batch_first(x: chex.Array) -> chex.Array:
    """
    Transpose from time-first to batch-first format after scan operations.

    Parameters
    ----------
    x : jax.Array
        Input tensor in time-first format `(T, B, ...)`

    Returns
    -------
    x : jax.Array
        Output tensor in batch-first format `(B, T, ...)`
    """
    # (T, B, ...) -> (B, T, ...)
    return jnp.swapaxes(x, 0, 1)
