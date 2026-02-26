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

import gymnasium as gym
import numpy as np


class FrameStackReshape(gym.ObservationWrapper):
    """
    Moves the frame stack dimension to the channel dimension.

    Gymnasium's `FrameStackObservation` wrapper stacks frames (FS) along a new
    leading axis after the batch, resulting in shape `(B, FS, H, W, C)`.
    This wrapper moves the frame stack to the channel dimension, producing
    `(B, H, W, FS)` which is the standard format for image observations.

    Note -
        Requires grayscale input with shape `(B, FS, H, W, 1)`.
        The trailing channel dimension is squeezed out and replaced by
        the frame stack dimension.

    Parameters
    ----------
    env : gym.Env
        The Gymnasium environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        # Update observation space to reflect new shape
        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)

        # (FS, H, W, 1) -> (H, W, FS)
        old_shape = old_space.shape
        assert old_shape is not None and len(old_shape) == 4

        FS, H, W, C = old_shape
        assert C == 1, f"Expected grayscale input with C=1, got C={C}"

        new_shape = (H, W, FS)

        self.observation_space = gym.spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=new_shape,
            dtype=old_space.dtype,  # type: ignore
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Transform observation from `(FS, H, W, 1)` to `(H, W, FS)`.

        Parameters
        ----------
        observation : np.ndarray
            Observation with shape `(FS, H, W, 1)`

        Returns
        -------
        obs : np.ndarray
            Observation with shape `(H, W, FS)`
        """
        # (FS, H, W, 1) -> (FS, H, W) -> (H, W, FS)
        obs = np.squeeze(observation, axis=-1)
        return np.transpose(obs, (1, 2, 0))
