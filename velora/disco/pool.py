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

from typing import Dict

import jax
import jax.numpy as jnp

from velora.disco.config.metadata import PoolMetadata


def pack_obs(obs_dict: Dict[str, jax.Array], meta: PoolMetadata) -> jax.Array:
    """
    Packs and pads a set of `envrax.MultiEnv` observations into the shape
    `(P, 1, max_obs_dim)`.

    Parameters
    ----------
    obs_dict : Dict[str, jax.Array]
        The multi-environment observation set
    meta : PoolMetadata
        Training pool metadata

    Returns
    -------
    obs : jax.Array
        Updated set of observations in the shape `(P, 1, max_obs_dim)`
    """
    padded = [
        jnp.pad(obs_dict[s.env_key], (0, meta.max_obs_dim - s.obs_dim))
        for s in meta.slots
    ]
    return jnp.stack(padded, axis=0)[:, None, :]


def unpack_actions(actions: jax.Array, meta: PoolMetadata) -> Dict[str, jax.Array]:
    """
    Unpacks a set of actions back into their multi-environment format.

    Parameters
    ----------
    actions : jax.Array
        Agent actions
    meta : PoolMetadata
        Trainer pool metadata

    Returns
    -------
    actions : Dict[str, jax.Array]
        Unpackaged actions key-paired by their slot environment key
    """
    return {s.env_key: actions[i, 0, s.action_dim] for i, s in enumerate(meta.slots)}
