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
import jax
import jax.numpy as jnp
from flax import nnx

from velora.disco.config.settings import DiscoEncoderSettings
from velora.disco.rollouts import Rollout
from velora.utils.nn import active_parameters, total_parameters


class DiscoInputEncoder(nnx.Module):
    """
    Encodes continuous policy agent predictions and environment signals into
    a readable format for the Disco network.

    Transforms raw inputs into fixed-size embeddings via single-layer projections:
        - State-conditional `(y, y_target)` → Linear projection
        - Policy encoder - encodes Gaussian policy parameters
          `(μ, log σ, μ_target, log σ_target)` into a policy summary embedding
        - Action-conditional encoder - encodes
          `(z, q, z_target, q_target, action_taken)` into an
          action-conditional embedding
        - Scalars `(rewards, discounts)` → Linear projection

    Parameters
    ----------
    config : DiscoEncoderSettings
        Encoder configuration
    max_action_dim : int
        Maximum continuous action dimensionality across all environments.
        Used to size the policy and action-conditional encoder inputs.
    rngs : nnx.Rngs
        Random number generator
    """

    def __init__(
        self,
        config: DiscoEncoderSettings,
        max_action_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.rngs = rngs

        self.max_action_dim = max_action_dim

        # (B, T, Y) -> (B, T, E_y)
        self.state_encoder = nnx.Linear(
            self.config.prediction_size,
            self.config.obs_embed_dim,
            rngs=rngs,
        )

        # (B, T, 2) -> (B, T, E_s)
        self.scalar_encoder = nnx.Linear(
            2,  # reward + discount
            self.config.scalar_embed_dim,
            rngs=rngs,
        )

        # Policy summary: (mu, log_std, mu_target, log_std_target) -> embed
        self.policy_encoder = nnx.Linear(
            4 * max_action_dim,
            config.action_embed_dim,
            rngs=rngs,
        )

        # Action-conditional: (z, q, z_target, q_target, action) -> embed
        # z: prediction_size, q: 1 (scalar), action: max_action_dim
        self.action_cond_encoder = nnx.Linear(
            2 * (config.prediction_size + 1) + max_action_dim,
            config.action_embed_dim,
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

    def _encode(self, x: jax.Array) -> jax.Array:
        """
        Encodes an array using Softmax.

        Parameters
        ----------
        x : jax.Array
            Input to encode in shape `(B, T, F)`

        Returns
        -------
        embedding : jax.Array
            Output embedding in shape: `(B, T, E)`
        """
        return nnx.softmax(x, axis=-1)

    def __call__(self, rollout: Rollout) -> jax.Array:
        """
        Performs a forward pass through the encoding.

        Parameters
        ----------
        rollout : Rollout
            A trajectory of experience containing:
                - `actions`: Continuous actions taken. Shape: `(B, T, D)`
                - `rewards`: Rewards received. Shape: `(B, T, 1)`
                - `discounts`: Episode continuation signals. Shape: `(B, T, 1)`
                - `preds`: policy network predictions
                - `target_preds`: target network predictions

        Returns
        -------
        embedding : jax.Array
            Flattened embedding for the Disco network. Shape: `(B, T, E)`
            where `E = output_dim` (same as discrete variant).
        """
        # State encoding — inherited from parent
        y, y_target = self._encode_states(rollout.preds.y, rollout.target_preds.y)

        # Policy summary encoding — Gaussian parameters
        policy_emb = self._encode_policy(
            rollout.preds.mu,
            rollout.preds.log_std,
            rollout.target_preds.mu,
            rollout.target_preds.log_std,
        )

        # Action-conditional encoding — z, q conditioned on taken action
        action_cond_emb = self._encode_action_conditional(
            rollout.preds.z,
            rollout.preds.q,
            rollout.target_preds.z,
            rollout.target_preds.q,
            rollout.actions,
        )

        # Scalar encoding — inherited from parent
        scalar = self._encode_scalars(rollout.rewards, rollout.discounts)

        # Same structure as discrete: (y, y_target, policy_summary, action_cond, scalar)
        # Matches output_dim = obs_embed_dim * 2 + action_embed_dim * 2 + scalar_embed_dim
        embedding = jnp.concatenate(
            [y, y_target, policy_emb, action_cond_emb, scalar],
            axis=-1,
        )

        return embedding

    def _encode_states(
        self,
        y: jax.Array,
        y_target: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Encode state-conditional predictions.

        Parameters
        ----------
        y : jax.Array
            Observation-conditioned predictions. Shape: `(B, T, Y)`
        y_target : jax.Array
            Target observation-conditioned predictions. Shape: `(B, T, Y)`

        Returns
        -------
        y_embed : jax.Array
            Y State embedding. Shape: `(B, T, E_y)`
        y_target_embed : jax.Array
            Y target state embedding. Shape: `(B, T, E_y)`
        """
        y_embed = self.state_encoder(self._encode(y))  # type: ignore
        y_target_embed = self.state_encoder(self._encode(y_target))  # type: ignore
        return y_embed, y_target_embed

    def _encode_scalars(self, rewards: jax.Array, discounts: jax.Array) -> jax.Array:
        """
        Encode scalar inputs (rewards, discounts).

        Applies sign-preserving log transform to rewards before encoding:
        `sign(r) * log(1 + |r|)`.

        Parameters
        ----------
        rewards : jax.Array
            Rewards. Shape: `(B, T, 1)`
        discounts : jax.Array
            Discounts. Shape: `(B, T, 1)`

        Returns
        -------
        scalar_emb : jax.Array
            Scalar embedding. Shape: `(B, T, E_s)`
        """
        rewards = jnp.squeeze(rewards, axis=-1)  # (B, T)
        discounts = jnp.squeeze(discounts, axis=-1)  # (B, T)

        rewards = jnp.sign(rewards) * jnp.log1p(jnp.abs(rewards) + 1e-3)
        scalars = jnp.stack([rewards, discounts], axis=-1)
        return self.scalar_encoder(scalars)

    def _encode_policy(
        self,
        mu: jax.Array,
        log_std: jax.Array,
        mu_target: jax.Array,
        log_std_target: jax.Array,
    ) -> jax.Array:
        """
        Encode Gaussian policy parameters into a summary embedding.

        Provides the meta-network with a view of the full policy distribution,
        analogous to the average-over-all-actions embedding in discrete.

        Parameters
        ----------
        mu : jax.Array
            Policy mean. Shape: `(B, T, D)`
        log_std : jax.Array
            Policy log standard deviation. Shape: `(B, T, D)`
        mu_target : jax.Array
            Target policy mean. Shape: `(B, T, D)`
        log_std_target : jax.Array
            Target policy log standard deviation. Shape: `(B, T, D)`

        Returns
        -------
        policy_emb : jax.Array
            Policy summary embedding. Shape: `(B, T, action_embed_dim)`
        """
        policy_input = jnp.concatenate(
            [mu, log_std, mu_target, log_std_target],
            axis=-1,
        )  # (B, T, 4 * max_action_dim)

        return self.policy_encoder(policy_input)

    def _encode_action_conditional(
        self,
        z: jax.Array,
        q: jax.Array,
        z_target: jax.Array,
        q_target: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """
        Encode action-conditional predictions and the taken action.

        Since `z` and `q` are already conditioned on the taken action
        (no action dimension), we concatenate them directly with the
        continuous action vector.

        Parameters
        ----------
        z : jax.Array
            Action-conditioned predictions. Shape: `(B, T, Z)`
        q : jax.Array
            Scalar Q-value. Shape: `(B, T, 1)`
        z_target : jax.Array
            Target z predictions. Shape: `(B, T, Z)`
        q_target : jax.Array
            Target Q-value. Shape: `(B, T, 1)`
        actions : jax.Array
            Continuous action taken. Shape: `(B, T, D)`

        Returns
        -------
        action_cond_emb : jax.Array
            Action-conditional embedding. Shape: `(B, T, action_embed_dim)`
        """
        action_cond_input = jnp.concatenate(
            [
                self._encode(z),  # softmax(z) — learned representation
                q,  # scalar Q — no softmax
                self._encode(z_target),
                q_target,
                actions,  # continuous action taken
            ],
            axis=-1,
        )  # (B, T, 2 * prediction_size + 2 + max_action_dim)

        return self.action_cond_encoder(action_cond_input)


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

        base_scale = 16
        scale_factor = 20
        output_scale = 64

        # Compute convolution dimensions dynamically based on n_hidden
        self.base_channels = max(base_scale, (n_hidden // scale_factor) * base_scale)
        self.output_dim = max(output_scale, (n_hidden // scale_factor) * output_scale)

        rngs = nnx.Rngs(params=key)

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

    def __call__(self, x: jax.Array) -> jax.Array:
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
    max_obs_dim : int (optional)
        When provided, the projection layer input width is set to
        `max_obs_dim` instead of `obs_dim`. Default is `None`
    """

    def __init__(
        self,
        obs_dim: int,
        n_hidden: int,
        key: chex.PRNGKey,
        max_obs_dim: int | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.max_obs_dim = max_obs_dim or obs_dim
        self.n_hidden = n_hidden

        # Dynamic encoder scaling for output
        output_scale = 64
        scale_factor = 20
        self.output_dim = max(output_scale, (n_hidden // scale_factor) * output_scale)

        rngs = nnx.Rngs(params=key)

        self.proj = nnx.Linear(self.max_obs_dim, self.output_dim, rngs=rngs)

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

    @property
    def encoding_dim(self) -> int:
        """Output feature dimensionality of the encoder."""
        return self.output_dim

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the encoder.

        Accepts either 2D `(B, D)` or 3D `(B, T, D)` input and returns
        encoded features in `(B, T, F)` format.

        Parameters
        ----------
        x : jax.Array
            Batch of vector observations:
            - `(B, D)` — single timestep
            - `(B, T, D)` — sequence

        Returns
        -------
        features : jax.Array
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
    obs_shape: Tuple[int, ...],
    n_hidden: int,
    key: chex.PRNGKey,
    max_obs_dim: int | None = None,
) -> PolicyEncoder:
    """
    Dynamically build an observation encoder based on an observation space's shape.

    Parameters
    ----------
    obs_shape : Tuple[int, ...]
        Observation space shape
    n_hidden : int
        Number of hidden units for scaling
    key : chex.PRNGKey
        Random number generator key
    max_obs_dim : int (optional)
        Maximum observation dimensionality for vector obs. When provided,
        the encoder's input layer is sized to `max_obs_dim` so that all
        encoders share the same weight shape (required for `jax.vmap`).
        Smaller observations are zero-padded to this size.
        Only used for 1D vector observations. Default is `None`

    Returns
    -------
    encoder : ImageEncoder | VectorEncoder
        Image encoder for 3D obs `(H, W, C)`, vector encoder for 1D obs `(D,)`
    """
    if len(obs_shape) == 3:
        # Image observations: (H, W, C)
        return ImageEncoder(obs_shape[-1], n_hidden, key=key)
    elif len(obs_shape) == 1:
        # Vector observations: (D,)
        return VectorEncoder(
            obs_shape[0],
            n_hidden,
            key=key,
            max_obs_dim=max_obs_dim,
        )
    else:
        raise ValueError(
            f"Unsupported observation shape: {obs_shape}. "
            "Expected 1D vector '(D,)' or 3D image '(H, W, C)'."
        )
