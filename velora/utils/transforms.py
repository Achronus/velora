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


def squeeze_time(x: torch.Tensor) -> torch.Tensor:
    """
    Squeeze the time dimension (axis 1) if `T=1`.

    Useful for removing redundant sequence dimensions when
    processing single timestep observations.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape `(B, T, ...)` where T may be 1

    Returns
    -------
    x : torch.Tensor
        Output tensor with shape `(B, ...)` if `T=1`, otherwise unchanged
    """
    if x.ndim >= 2 and x.shape[1] == 1:
        return torch.squeeze(x, dim=1)

    return x
