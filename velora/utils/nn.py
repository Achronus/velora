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

import torch
from torch import nn


def total_parameters(model: nn.Module) -> int:
    """
    Calculates the total number of parameters used in a `nn.Module`.

    Parameters
    ----------
    model : nn.Module
        A module with parameters

    Returns
    -------
    count : int
        The total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def active_parameters(model: nn.Module) -> int:
    """
    Calculates the active number of parameters used in a `nn.Module`.

    Filters out parameters that are `0`.

    Parameters
    ----------
    model : nn.Module
        A module with parameters

    Returns
    -------
    count : int
        The total active number of parameters
    """
    return sum(int(torch.count_nonzero(p)) for p in model.parameters())
