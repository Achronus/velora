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

import jax.numpy as jnp
from flax import struct

from velora.core.distributions import CategoricalBins


@struct.dataclass(frozen=True)
class AgentSettings:
    """
    Dataclass for `VeloraAgent` settings.

    Parameters
    ----------
    n_hidden : int
        Number of decision nodes for policy networks (inter + command nodes)
    prediction_size : int
        Size of the observation/action-conditioned prediction vectors (y, z)
    q_size : int
        Size of action-value prediction head. Controls the number of discrete bins
        for distributional Q-values (`n_atoms`)
    lr : float (optional)
        Learning rate for the agent's optimizer. Default is `0.0005`
    max_grad_norm : float (optional)
        Maximum gradient norm for gradient clipping. Default is `1.0`
    bin_resolution : float (optional)
        Target bin resolution (width) for distributional Q-values. Dynamically
        sets the range of Q-values that can be represented for `[min, max]` based
        on `q_size`. Smaller resolutions provide finer granularity for value
        predictions but reduce the representable range. Default is `0.4`
    sparsity_level : float (optional)
        Network connection sparsity between neurons used for the Liquid Neural
        Networks (LNNs). Default is `0.5`
    """

    n_hidden: int
    prediction_size: int
    q_size: int
    lr: float = 5e-4
    max_grad_norm: float = 1.0
    bin_resolution: float = 0.4
    sparsity_level: float = 0.5

    def categorical_bins(self) -> CategoricalBins:
        """
        Computes the categorical bin values for distributional Q-values.

        Returns
        -------
        bins : CategoricalBins
            An object containing the categorical bin values
        """
        max_bin_value = (self.bin_resolution * (self.q_size - 1)) / 2.0
        support = jnp.linspace(-max_bin_value, max_bin_value, num=self.q_size)
        return CategoricalBins(
            support=support,
            min_value=-max_bin_value,
            max_value=max_bin_value,
        )
