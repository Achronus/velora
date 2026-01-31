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

from velora.config.outputs import BufferSamples, PolicyAgentOutput
from velora.config.settings import DiscoEncoderSettings


class DiscoInputEncoder(nnx.Module):
    """
    Encodes the policy agent predictions and environment signals into a readable format for the Disco network.

    Transforms raw inputs into fixed-size embeddings via single-layer projections:
        - State-conditional `(y, y_target)` → Linear projection
        - Action-conditional `(z, q, pi)` → Shared Linear projection across actions
        - Scalars `(rewards, discounts)` → Linear projection

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
        self.n_actions = self.config.n_actions

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

    def __call__(self, samples: BufferSamples) -> Tuple[chex.Array, chex.Array]:
        """
        Performs a forward pass through the encoding.

        Parameters
        ----------
        samples : BufferSamples
            Batch of samples from the buffer containing:
                - `actions`: Actions taken. Shape: `(B, T)`
                - `rewards`: Rewards received. Shape: `(B, T)`
                - `discounts`: Episode continuation signals. Shape: `(B, T)`
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
        y, y_target = self._encode_states(samples.preds.y, samples.target_preds.y)

        action_emb, action_avg, action_a = self._encode_actions(
            samples.preds,
            samples.target_preds,
            samples.actions,
        )

        scalar = self._encode_scalars(samples.rewards, samples.discounts)

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
        # Policy probabilities per action: (B, T, A) -> (B, T, A, 1)
        pi_probs = jnp.expand_dims(self._encode(preds.pi), axis=-1)
        pi_target_probs = jnp.expand_dims(self._encode(targets.pi), axis=-1)

        # One-hot encode actions taken: (B, T) -> (B, T, A) -> (B, T, A, 1)
        actions = jnp.squeeze(actions, axis=-1)  # (B, T)
        one_hot_actions = jax.nn.one_hot(actions, self.n_actions)
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
