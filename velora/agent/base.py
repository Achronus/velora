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


import random
from abc import ABC, abstractmethod

import numpy as np
import torch

from velora.utils.nn import set_torch_device


class RLAgent(ABC):
    """
    An abstract base class for all Reinforcement Learning agents.

    Parameters
    ----------
    seed : int
        Random number generator seed
    device : torch.device (optional)
        Device to load tensors onto. When `None`, sets device to CUDA
        or CPU automatically. Default is `None`
    """

    def __init__(self, *, seed: int, device: torch.device | None) -> None:
        self.seed = seed
        self.device = device if device is not None else set_torch_device()

        # Configure randomness
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("high")

    @abstractmethod
    def train(self) -> None:
        """Trains the agent."""
        ...
