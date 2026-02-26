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

from typing import Tuple

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from velora.disco.config.settings import DiscoEncoderSettings
from velora.disco.outputs import PolicyAgentOutput
from velora.disco.rollouts import Rollout
from velora.utils.nn import active_parameters, total_parameters


class DiscoInputEncoder(nnx.Module):
    """
    Encodes the policy agent predictions and environment signals into a readable format for the Disco network.

    Transforms raw inputs into fixed-size embeddings via single-layer projections:
        - State-conditional `(y, y_target)` → Linear projection
        - Action-conditional `(z, q, pi)` → Shared Linear projection across actions
        - Scalars `(rewards, discounts)` → Linear projection

    The encoder derives `n_actions` from input shapes at runtime, enabling
    action-agnostic encoding that works across different action spaces.

    Parameters
    ----------
    config : DiscoEncoderSettings
        Encoder configuration
    rngs : nnx.Rngs
        Random number generator
    """

    def __init__(self, *, config: DiscoEncoderSettings, rngs: nnx.Rngs) -> None:
        self.config = config

        self.out_features = self.config.output_dim

        # (B, T, Y) -> (B, T, E_y)
        self.state_encoder = nnx.Linear(
            self.config.prediction_size,
            self.config.obs_embed_dim,
            rngs=rngs,
        )

        # (B, T, A, Z+Q+2) -> (B, T, A, E_a)
        self.action_encoder = nnx.Linear(
            self.config.action_input_dim,
            self.config.action_embed_dim,
            rngs=rngs,
        )

        # (B, T, 2) -> (B, T, E_s)
        self.scalar_encoder = nnx.Linear(
            2,  # reward + discount
            self.config.scalar_embed_dim,
            rngs=rngs,
        )

        self._total_params = total_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns
        -------
        count : int
            The total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns
        -------
        count : int
            The active parameter count.
        """
        return self._total_params

    def __call__(self, rollout: Rollout) -> Tuple[chex.Array, chex.Array]:
        """
        Performs a forward pass through the encoding.

        Parameters
        ----------
        rollout : Rollout
            A trajectory of experience containing:
                - `actions`: Actions taken. Shape: `(B, T, 1)`
                - `rewards`: Rewards received. Shape: `(B, T, 1)`
                - `discounts`: Episode continuation signals. Shape: `(B, T, 1)`
                - `values` : State value estimates. Shape `(B, T, 1)`
                - `preds`: Agent predictions
                - `target_preds`: Target network predictions

        Returns
        -------
        embedding : chex.Array
            Flattened embedding for the Disco network. Shape: `(B, T, E)`

                - batch_size (`B`) - the number of samples per timestep.
                - seq_length (`T`) - the number of sequences (e.g., trajectories).
                - output_dim (`E`) - the combined embedding dimension.

        action_embedding : chex.Array
            Per-action embeddings for the policy target decoding. Shape: `(B, T, A, C)`

                - batch_size (`B`) - the number of samples per timestep.
                - seq_length (`T`) - the number of sequences (e.g., trajectories).
                - n_actions (`A`) - the number of discrete actions in the action space.
                - action_embed_dim (`C`) - the action embedding dimension.
        """
        y, y_target = self._encode_states(rollout.preds.y, rollout.target_preds.y)

        action_emb, action_avg, action_a = self._encode_actions(
            rollout.preds,
            rollout.target_preds,
            rollout.actions,
        )

        scalar = self._encode_scalars(rollout.rewards, rollout.discounts)

        embedding = jnp.concatenate(
            [y, y_target, action_avg, action_a, scalar],
            axis=-1,
        )

        return embedding, action_emb

    def _encode(self, x: chex.Array) -> chex.Array:
        """
        Encodes an array using Softmax.

        Parameters
        ----------
        x : chex.Array
            Input to encode in shape `(B, T, F)`

        Returns
        -------
        embedding : chex.Array
            Output embedding in shape: `(B, T, E)`
        """
        return nnx.softmax(x, axis=-1)

    def _encode_states(
        self, y: chex.Array, y_target: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Encode state-conditional predictions.

        Parameters
        ----------
        y : chex.Array
            Observation-conditioned predictions. Shape: `(B, T, Y)`
        y_target : chex.Array
            Target observation-conditioned predictions. Shape: `(B, T, Y)`

        Returns
        -------
        y_embed : chex.Array
            Y State embedding. Shape: `(B, T, E_y)`
        y_target_embed : chex.Array
            Y target state embedding. Shape: `(B, T, E_y)`
        """
        y_embed = self.state_encoder(self._encode(y))  # type: ignore
        y_target_embed = self.state_encoder(self._encode(y_target))  # type: ignore
        return y_embed, y_target_embed

    def _encode_actions(
        self,
        preds: PolicyAgentOutput,
        targets: PolicyAgentOutput,
        actions: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Encode action-conditional inputs with shared weights across actions.

        Concatenates `(z, q, pi, target_z, target_q, target_pi, one_hot_action)` for each action, then projects them through a shared Linear layer.

        Derives `n_actions` from input shapes at runtime.

        Parameters
        ----------
        preds : PolicyAgentOutput
            Policy agent network output predictions
        targets : PolicyAgentOutput
            Policy agent target network output predictions
        actions : jax.Array
            Actions taken with shape `(B, T, 1)`:

            - batch_size (`B`) - the number of samples per timestep.
            - seq_length (`T`) - the number of timesteps in the trajectory.

        Returns
        -------
        action_embedding : chex.Array
            Per-action embeddings. Shape: `(B, T, A, C)`
        action_emb_avg : chex.Array
            Average across actions. Shape: `(B, T, C)`
        action_emb_a : chex.Array
            Embedding for action taken. Shape: `(B, T, C)`
        """
        # Derive n_actions from input shape (z has shape (B, T, A, Z))
        n_actions = jnp.shape(preds.z)[2]

        # Policy probabilities per action: (B, T, A) -> (B, T, A, 1)
        pi_probs = jnp.expand_dims(self._encode(preds.pi), axis=-1)
        pi_target_probs = jnp.expand_dims(self._encode(targets.pi), axis=-1)

        # One-hot encode actions taken: (B, T) -> (B, T, A) -> (B, T, A, 1)
        actions = jnp.squeeze(actions, axis=-1)  # (B, T)
        one_hot_actions = jax.nn.one_hot(actions, n_actions)
        one_hot_actions = jnp.expand_dims(one_hot_actions, axis=-1)

        # Merge action-conditional features: (B, T, A, 2*(Z+Q+1)+1)
        action_inputs = jnp.concatenate(
            [
                self._encode(preds.z),
                self._encode(preds.q),
                pi_probs,
                self._encode(targets.z),
                self._encode(targets.q),
                pi_target_probs,
                one_hot_actions,
            ],
            axis=-1,
        )

        # Compute outputs
        action_embedding = self.action_encoder(action_inputs)  # (B, T, A, C)
        action_emb_avg = jnp.mean(action_embedding, axis=2)  # (B, T, C)

        # Select embedding for action taken
        idx = jnp.expand_dims(actions, axis=(2, 3))  # (B, T, 1, 1)
        action_emb_a = jnp.take_along_axis(action_embedding, idx, axis=2).squeeze(
            axis=2
        )  # (B, T, 1, C) -> (B, T, C)

        return action_embedding, action_emb_avg, action_emb_a

    def _encode_scalars(self, rewards: chex.Array, discounts: chex.Array) -> chex.Array:
        """
        Encode scalar inputs (rewards, discounts).

        Applies sign-preserving log transform to rewards before encoding: `sign(r) * log(1 + |r|)`.

        Parameters
        ----------
        rewards : chex.Array
            Rewards. Shape: `(B, T, 1)`
        discounts : chex.Array
            Discounts. Shape: `(B, T, 1)`

        Returns
        -------
        scalar_emb : chex.Array
            Scalar embedding. Shape: `(B, T, E_s)`
        """
        rewards = jnp.squeeze(rewards, axis=-1)  # (B, T)
        discounts = jnp.squeeze(discounts, axis=-1)  # (B, T)

        rewards = jnp.sign(rewards) * jnp.log1p(jnp.abs(rewards) + 1e-3)
        scalars = jnp.stack([rewards, discounts], axis=-1)
        return self.scalar_encoder(scalars)


class ImageEncoder(nnx.Module):
    """
    A simple Convolutional Neural Network (CNN) used for encoding images.

    Dynamically scales convolutional layer dimensions based on the number of
    architecture hidden units.

    Architecture:
        Conv1: base_channels, 5x5 kernel, stride 2, padding 2, ReLU
        Conv2: base_channels * 2, 5x5 kernel, stride 2, padding 2, ReLU
        Global Average Pooling
        Linear: base_channels * 2, output_dim, ReLU

    Scaling is determined by:
        ```python
        base_channels = max(16, (n_hidden // 20) * 16)
        output_dim = max(64, (n_hidden // 20) * 64)
        ```

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 4 for frame-stacked grayscale)
    n_hidden : int
        Number of hidden units. Used to compute convolution dimensions
    key : jax.random.PRNGKey
        Random number generator key
    """

    def __init__(self, in_channels: int, n_hidden: int, key: chex.PRNGKey) -> None:
        self.in_channels = in_channels
        self.n_hidden = n_hidden
        self.key = key

        base_scale = 16
        scale_factor = 20
        output_scale = 64

        # Compute convolution dimensions dynamically based on n_hidden
        self.base_channels = max(base_scale, (n_hidden // scale_factor) * base_scale)
        self.output_dim = max(output_scale, (n_hidden // scale_factor) * output_scale)

        rngs = nnx.Rngs(params=self.key)

        self.conv1 = nnx.Conv(
            self.in_channels,
            self.base_channels,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding=2,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            self.base_channels,
            self.base_channels * 2,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding=2,
            rngs=rngs,
        )

        # Output projection
        self.proj = nnx.Linear(
            self.base_channels * 2,
            self.output_dim,
            rngs=rngs,
        )

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """
        Gets the network's total parameter count.

        Returns
        -------
        count : int
            The total parameter count.
        """
        return self._total_params

    @property
    def active_params(self) -> int:
        """
        Gets the network's active parameter count.

        Returns
        -------
        count : int
            The active parameter count.
        """
        return self._active_params

    @property
    def encoding_dim(self) -> int:
        """Output feature dimensionality of the encoder."""
        return self.output_dim

    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Perform a forward pass through the network.

        Accepts either 4D or 5D input and returns encoded features
        in `(B, T, F)` format.

        Parameters
        ----------
        x : jax.Array
            Batch of images in shape `(B, H, W, C)`
            or `(B, T, H, W, C)`

            - `batch_size (B)` the number of images
            - `seq_length (T)` the number of timesteps
            - `height (H)` the height of each image
            - `width (W)` the width of each image
            - `channels (C)` the number of channels per image

        Returns
        -------
        features : jax.Array
            Encoded features in `(B, T, F)` format

            - `batch_size (B)` the number of samples
            - `seq_length (T)` the number of timesteps (1 if 4D input)
            - `features (F)` the output feature dimension
        """
        if x.ndim not in [4, 5]:
            raise ValueError(
                f"'x' must be shape '(B, H, W, C)' or '(B, T, H, W, C)'. Got: x={jnp.shape(x)}"
            )

        if x.ndim == 4:
            x = jnp.expand_dims(x, axis=1)  # (B, H, W, C) -> (B, 1, H, W, C)

        B, T, H, W, C = jnp.shape(x)

        # Flatten batch and time: (B, T, H, W, C) -> (B*T, H, W, C)
        x_flat = jnp.reshape(x, shape=(B * T, H, W, C))

        # Encode through CNN
        x = nnx.relu(self.conv1(x_flat))
        x = nnx.relu(self.conv2(x))
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling -> (B, C * 2)
        x = nnx.relu(self.proj(x))  # (B, output_dim)

        # Reshape back: (B*T, F) -> (B, T, F)
        return x.reshape(B, T, -1)


class VectorEncoder(nnx.Module):
    """
    A simple Linear encoder for flat vector observations (e.g., MuJoCo action space).

    Dynamically scales the output dimension based on the number of provided hidden units.

    Can act as a drop-in-replacement to `ImageEncoder` when the observation space is
    a 1D vector rather than an image.

    Architecture:
        Linear: observation vector projection to output `(B, T, F)`
        ReLU: activation function

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector
    n_hidden : int
        Number of hidden units
    key : chex.PRNGKey
        Random number generator key
    """

    def __init__(self, obs_dim: int, n_hidden: int, key: chex.PRNGKey) -> None:
        self.obs_dim = obs_dim
        self.n_hidden = n_hidden
        self.key = key

        # Dynamic encoder scaling for output
        output_scale = 64
        scale_factor = 20
        self.output_dim = max(output_scale, (n_hidden // scale_factor) * output_scale)

        rngs = nnx.Rngs(params=self.key)

        self.proj = nnx.Linear(self.obs_dim, self.output_dim, rngs=rngs)

        self._total_params = total_parameters(self)
        self._active_params = active_parameters(self)

    @property
    def total_params(self) -> int:
        """Total parameter count."""
        return self._total_params

    @property
    def active_params(self) -> int:
        """Active parameter count."""
        return self._active_params

    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Forward pass through the encoder.

        Accepts either 2D `(B, D)` or 3D `(B, T, D)` input and returns
        encoded features in `(B, T, F)` format.

        Parameters
        ----------
        x : chex.Array
            Batch of vector observations:
            - `(B, D)` — single timestep
            - `(B, T, D)` — sequence

        Returns
        -------
        features : chex.Array
            Encoded features `(B, T, F)`
        """
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=1)  # (B, D) -> (B, 1, D)

        B, T, D = jnp.shape(x)

        # Flatten batch and time for MLP: (B*T, D)
        x_flat = jnp.reshape(x, (B * T, D))
        x_flat = nnx.relu(self.proj(x_flat))

        return x_flat.reshape(B, T, -1)  # (B, T, F)


PolicyEncoder = ImageEncoder | VectorEncoder


def build_obs_encoder(
    obs_spec: gym.spaces.Box,
    n_hidden: int,
    key: chex.PRNGKey,
) -> PolicyEncoder:
    """
    Dynamically build an observation encoder based on `obs_spec` shape.

    Parameters
    ----------
    obs_spec : gym.spaces.Box
        Observation space specification
    n_hidden : int
        Number of hidden units for scaling
    key : chex.PRNGKey
        Random number generator key

    Returns
    -------
    encoder : ImageEncoder | VectorEncoder
        Image encoder for 3D obs `(H, W, C)`, vector encoder for 1D obs `(D,)`
    """
    obs_shape = obs_spec.shape

    if len(obs_shape) == 3:
        # Image observations: (H, W, C)
        return ImageEncoder(obs_shape[-1], n_hidden, key=key)
    elif len(obs_shape) == 1:
        # Vector observations: (D,)
        return VectorEncoder(obs_shape[0], n_hidden, key=key)
    else:
        raise ValueError(
            f"Unsupported observation shape: {obs_shape}. "
            "Expected 1D vector '(D,)' or 3D image '(H, W, C)'."
        )
