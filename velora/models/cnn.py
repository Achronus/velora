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

import chex
import flax.nnx as nnx
import jax.numpy as jnp


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
        self.output_proj = nnx.Linear(
            self.base_channels * 2,
            self.output_dim,
            rngs=rngs,
        )

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
        x = nnx.relu(self.output_proj(x))  # (B, output_dim)

        # Reshape back: (B*T, F) -> (B, T, F)
        return x.reshape(B, T, -1)
