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

from typing import Optional

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax

from velora.config.outputs import AgentOutput
from velora.config.settings import AgentSettings
from velora.config.state import AgentHiddenStates
from velora.core.optim import scale_by_adam_no_denom
from velora.models.cnn import ImageEncoder
from velora.models.policy import ACM, OCM


class VeloraAgent:
    """
    Creates a policy agent used to discover Reinforcement Learning rules.
    Combines DiscoRL techniques with Liquid Neural Networks (LNNs).

    References:
        - [Discovering state-of-the-art reinforcement learning algorithms (2025)](https://www.nature.com/articles/s41586-025-09761-x)
        - [Liquid Time-constant Networks (2020)](https://arxiv.org/abs/2006.04439)

    Parameters:
        obs_spec (gym.spaces.Box): the observation space of the
            vectorized Gymnasium environment
        act_spec (gym.spaces.MultiDiscrete): the action space of the
            vectorized Gymnasium environment
        settings (AgentSettings): settings for the Velora agent
        seed (int, optional): random number generator seed.
            Default is `42`
    """

    def __init__(
        self,
        obs_spec: gym.spaces.Box,
        act_spec: gym.spaces.MultiDiscrete,
        settings: AgentSettings,
        *,
        seed: int = 42,
    ) -> None:
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.settings = settings
        self.seed = seed

        self.n_actions = self.act_spec.shape[0]

        key = jax.random.key(seed)
        key_cnn, key_ocm, key_acm = jax.random.split(key, 3)

        # Categorical bins for Q-values
        self.categorical_bins = settings.categorical_bins()

        self.cnn = ImageEncoder(
            obs_spec.shape[-1],
            settings.n_hidden,
            key=key_cnn,
        )

        self.ocm = OCM(
            self.cnn.output_dim,
            settings.n_hidden,
            settings.prediction_size,
            self.n_actions,
            key=key_ocm,
            sparsity_level=settings.sparsity_level,
        )

        self.acm = ACM(
            self.ocm.embedding_size,
            settings.n_hidden,
            settings.prediction_size,
            self.n_actions,
            self.categorical_bins.num_bins,
            key=key_acm,
            sparsity_level=settings.sparsity_level,
        )

        self.optimizer = optax.chain(
            scale_by_adam_no_denom(),
            optax.clip(settings.max_grad_norm),
            optax.scale(-settings.lr),
        )

    def __call__(
        self,
        obs: jax.Array,
        *,
        ocm_h_state: Optional[chex.Array] = None,
        acm_h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> AgentOutput:
        """
        Forward pass through the agent.

        Parameters:
            obs (jax.Array): image input observations `(B, T, H, W, C)`

                - `batch_size (B)` the number of samples per timestep.
                - `seq_length (T)` the number of sequences (e.g., trajectories).
                - `height (H)` the height of the image observation.
                - `width (W)` the width of the image observation.
                - `channels (C)` the number of channels in the image.

            ocm_h_state (jax.Array, optional): hidden state for the OCM
                `(B, HS)`. If `None`, initializes to zeros. Default is `None`

                - `batch_size (B)` the number of samples per timestep.
                - `n_units (HS)` the total number of OCM hidden neurons.

            acm_h_state (jax.Array, optional): hidden state for the ACM
                `(B*A, HS)`. If `None`, initializes to zeros. Default is `None`

                - `batch_size (B)` the number of samples per timestep.
                - `n_actions (A)` the number of discrete actions.
                - `n_units (HS)` the total number of ACM hidden neurons.

            timespans (jax.Array, optional): time intervals between
                observations `(T,)`. If `None`, uses uniform time intervals.
                Default is `None`

                - `seq_length (T)` the number of sequences (e.g., trajectories).

        Returns:
            preds (AgentOutput): an object of agent predictions.
        """
        B, T, H, W, C = jnp.shape(obs)

        # Encode all frames through CNN: (B, T, H, W, C) -> (B, T, F)
        obs_flat = obs.reshape(B * T, H, W, C)
        encoded = self.cnn(obs_flat)  # (B*T, F)
        encoded = encoded.reshape(B, T, -1)  # (B, T, F)

        # Transpose for OCM: (B, T, F) -> (B, F, T)
        encoded = jnp.transpose(encoded, (0, 2, 1))  # (B, F, T)

        # OCM forward - process observations and produce embeddings
        pi, y, embedding, ocm_h_state = self.ocm(
            encoded,
            h_state=ocm_h_state,
            timespans=timespans,
        )

        # ACM forward - use OCM embeddings for action-value predictions
        z, aux_pi, q, acm_h_state = self.acm(
            embedding,
            h_state=acm_h_state,
            timespans=timespans,
        )

        return AgentOutput(
            pi=pi,
            y=y,
            z=z,
            aux_pi=aux_pi,
            q=q,
            embedding=embedding,
            h_states=AgentHiddenStates(
                ocm=ocm_h_state,
                acm=acm_h_state,
            ),
        )
