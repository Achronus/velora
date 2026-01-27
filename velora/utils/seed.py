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

from typing import List

import chex
import jax
import jax.numpy as jnp


def restore_rng_key(x: List[int]) -> chex.PRNGKey:
    """
    Restores a `jax.random.key` from a set of key data (seed list).

    Parameters
    ----------
    x : List[int]
        Key data to store (seed list)

    Returns
    -------
    key : chex.PRNGKey
        Restored RNG key
    """
    return jax.random.wrap_key_data(jnp.array(x, dtype=jnp.uint32))


def get_rng_key_data(x: chex.PRNGKey) -> List[int]:
    """
    Extracts a `jax.random.key` data into a seed list.

    Parameters
    ----------
    x : chex.PRNGKey
        Key to extract

    Returns
    -------
    seed_list : List[int]
        An list of seed integers for the RNG key
    """
    return jax.random.key_data(x).tolist()
