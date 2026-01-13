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
from typing import Any, Dict, List, Tuple

import chex
import gymnasium as gym
import jax.numpy as jnp
import numpy as np


class JaxConversion(gym.vector.VectorWrapper):
    """
    Gymnasium Vector Environment wrapper that converts outputs to Jax arrays.

    Assumes vectorized environment uses the default standard (NumPy).

    Parameters
    ----------
    env : gymnasium.vector.VectorEnv
        Vectorized environment to convert
    """

    def __init__(self, env: gym.vector.VectorEnv) -> None:
        super().__init__(env)
        self.env = env

    def _to_jax(self, array: np.ndarray) -> chex.Array:
        """Convert NumPy array to Jax array."""
        return jnp.asarray(array)

    def _to_numpy(self, array: chex.Array) -> np.ndarray:
        """Convert Jax array to NumPy array."""
        return np.asarray(array)

    def reset(
        self,
        *,
        seed: int | List[int] | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[chex.Array, Dict[str, Any]]:
        """
        Reset all parallel environments and return a batch of initial
        observations and info.

        Parameters
        ----------
        seed : int (optional)
            The environment reset seed
        options : Dict[str, Any] (optional)
            Options to return

        Returns
        -------
        obs : chex.Array
            A batch of starting observations
        info : Dict[str, Any]
            Environment metadata
        """
        obs, info = self.env.reset(seed=seed, options=options)  # type: ignore
        return self._to_jax(obs), info

    def step(
        self, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Take an action for each parallel environment.

        Note: vectorized environments are reset automatically
        during `env.step()`.

        Parameters
        ----------
        action : jax.Array
            Batch of actions with the `action_space` shape

        Returns
        -------
        next_obs : jax.Array
            Batch of next observations
        reward : jax.Array
            Batch of rewards obtained
        done : jax.Array
            Batch of episode dones (terminations/truncations). Values:
                - `1.0` = episode continues
                - `0.0` = episode ended
        info : Dict[str, Any]
            Environment metadata
        """
        obs, reward, terminated, truncated, info = self.env.step(self._to_numpy(action))
        done = jnp.where(terminated | truncated, 0.0, 1.0).astype(jnp.float32)

        return self._to_jax(obs), self._to_jax(reward), done, info


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
            dtype=old_space.dtype,
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
