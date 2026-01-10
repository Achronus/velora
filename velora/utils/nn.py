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

import flax.nnx as nnx
import jax
import numpy as np


def total_parameters(model: nnx.Module) -> int:
    """
    Calculates the total number of parameters used in a Flax `nnx.Module`.

    Parameters
    ----------
    model : nnx.Module
        A Flax module with parameters

    Returns
    -------
    count : int
        The total number of parameters
    """
    params = nnx.state(model, nnx.Param)
    return np.sum([np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)])


def active_parameters(model: nnx.Module) -> int:
    """
    Calculates the active number of parameters used in a Flax `nnx.Module`.

    Filters out parameters that are `0`.

    Parameters
    ----------
    model : nnx.Module
        A Flax module with parameters

    Returns
    -------
    count : int
        The total active number of parameters
    """
    params = nnx.state(model, nnx.Param)
    return np.sum([p[(p != 0)].shape for p in jax.tree_util.tree_leaves(params)])
