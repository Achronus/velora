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


from typing import Any, Dict, Tuple

import torch
from gymnasium.wrappers.array_conversion import array_conversion
from gymnasium.wrappers.vector import NumpyToTorch


class NumpyToTorchRawInfo(NumpyToTorch):
    """
    A `NumpyToTorch` vector wrapper that leaves `info` dicts unconverted.

    Gymnasium's `NumpyToTorch` converts the entire `info` dict to
    PyTorch tensors, which fails on object arrays such as `final_obs`
    produced by same-step autoreset. This wrapper converts observations,
    rewards, terminations, truncations and actions as usual, but returns
    `info` as raw NumPy.
    """

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Performs a vectorized environment step, converting everything
        except `info` to PyTorch tensors.

        Parameters
        ----------
        actions : torch.Tensor
            A batch of agent actions `(n_envs, *act_shape)`

        Returns
        -------
        obs : torch.Tensor
            The next observations `(n_envs, *obs_shape)`
        rewards : torch.Tensor
            The rewards received `(n_envs,)`
        terminations : torch.Tensor
            The episode termination flags `(n_envs,)`
        truncations : torch.Tensor
            The episode truncation flags `(n_envs,)`
        info : Dict
            The raw, unconverted info dict
        """
        actions = array_conversion(actions, xp=self._env_xp, device=self._env_device)
        obs, reward, terminated, truncated, info = self.env.step(actions)

        return (
            array_conversion(obs, xp=self._target_xp, device=self._target_device),
            array_conversion(reward, xp=self._target_xp, device=self._target_device),
            array_conversion(
                terminated, xp=self._target_xp, device=self._target_device
            ),
            array_conversion(truncated, xp=self._target_xp, device=self._target_device),
            info,
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Resets the vectorized environments, converting only the
        observations to PyTorch tensors.

        Parameters
        ----------
        seed : int | list[int] (optional)
            The seed(s) for resetting the environments. Default is `None`
        options : Dict[str, Any] (optional)
            Additional reset options. Default is `None`

        Returns
        -------
        obs : torch.Tensor
            The initial observations `(n_envs, *obs_shape)`
        info : Dict
            The raw, unconverted info dict
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return (
            array_conversion(obs, xp=self._target_xp, device=self._target_device),
            info,
        )