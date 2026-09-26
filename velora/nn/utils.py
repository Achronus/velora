# Copyright 2026 Achronus
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

import numpy as np
import torch
from torch import nn


def layer_init(
    layer: nn.Linear,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear:
    """
    Initializes a linear layer in-place with orthogonal weights and
    constant bias.

    Parameters
    ----------
    layer : nn.Linear
        The linear layer to initialize
    std : float (optional)
        The gain (scaling factor) for the orthogonal weights.
        Default is `np.sqrt(2)`
    bias_const : float (optional)
        The constant value to fill the bias with. Default is `0.0`

    Returns
    -------
    layer : nn.Linear
        The initialized linear layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
