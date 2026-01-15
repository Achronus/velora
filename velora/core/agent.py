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

from typing import Optional, Tuple

import chex
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from velora.config.outputs import AgentOutput
from velora.config.settings import AgentSettings
from velora.config.state import AgentHiddenStates
from velora.core.optim import scale_by_adam_no_denom
from velora.models.cnn import ImageEncoder
from velora.models.policy import ACM, OCM


class PolicyAgent:
    """
    Creates a policy agent used to discover Reinforcement Learning (RL) rules.

    Combines DiscoRL techniques with Liquid Neural Networks (LNNs).

    Architecture:
        1. CNN Encoder - extracts visual features from image observations
        2. Observation-Conditional Model (OCM) - processes encoded features
           to produce policy logits (π) and observation-conditioned
           predictions (y)
        3. Action-Conditional Model (ACM) - takes OCM embeddings and produces
           action-conditioned predictions (z), auxiliary policy (aux_π),
           and Q-values (q) for all actions

    Output Shapes:
        - π (policy): `(B, A, T)`
        - y (obs-conditioned prediction): `(B, Y, T)`
        - z (action-conditioned prediction): `(B, A, Z, T)`
        - aux_π (auxiliary policy): `(B, A, A, T)`
        - q (action-values): `(B, A, Q, T)`

    References:
        - [Discovering state-of-the-art reinforcement learning algorithms (2025)](https://www.nature.com/articles/s41586-025-09761-x)
        - [Liquid Time-constant Networks (2020)](https://arxiv.org/abs/2006.04439)

    Parameters
    ----------
    obs_spec : gym.spaces.Box
        A single observation space of the vectorized Gymnasium environment
    act_spec : gym.spaces.Discrete
        A single action space of the vectorized Gymnasium environment
    config : AgentSettings
        Configuration for the policy agent
    key : jax.random.PRNGKey
        Random number generator key
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    """

    def __init__(
        self,
        obs_spec: gym.spaces.Box,
        act_spec: gym.spaces.Discrete,
        *,
        config: AgentSettings,
        key: chex.PRNGKey,
        jit_compile: bool = False,
    ) -> None:
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.config = config
        self.key = key

        self.n_actions = int(self.act_spec.n)

        key_cnn, key_ocm, key_acm, key_actions = jax.random.split(self.key, 4)
        self.key_actions = key_actions

        # Categorical bins for Q-values
        self.categorical_bins = config.categorical_bins()

        self.cnn = ImageEncoder(
            obs_spec.shape[-1],
            config.n_hidden,
            key=key_cnn,
        )

        self.ocm = OCM(
            self.cnn.output_dim,
            config.n_hidden,
            config.prediction_size,
            self.n_actions,
            key=key_ocm,
            sparsity=config.sparsity,
        )

        self.acm = ACM(
            self.ocm.embedding_size,
            config.n_hidden,
            config.prediction_size,
            self.n_actions,
            self.categorical_bins.num_bins,
            key=key_acm,
            sparsity=config.sparsity,
        )

        self.optimizer = optax.chain(
            scale_by_adam_no_denom(),
            optax.clip(config.max_grad_norm),
            optax.scale(-config.lr),
        )

        if jit_compile:
            self.cnn = nnx.jit(self.cnn)
            self.ocm = nnx.jit(self.ocm)
            self.acm = nnx.jit(self.acm)

    def __call__(
        self,
        obs: chex.Array,
        *,
        ocm_h_state: Optional[chex.Array] = None,
        acm_h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[AgentOutput, AgentHiddenStates]:
        """
        Forward pass through the agent.

        Parameters
        ----------
        obs : jax.Array
            Image input observations `(B, H, W, C)` or `(B, T, H, W, C)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories)
            - `height (H)` the height of the image observation
            - `width (W)` the width of the image observation
            - `channels (C)` the number of channels in the image

        ocm_h_state : jax.Array (optional)
            Hidden state for the OCM `(B, HS)`. If `None`, initializes to zeros.
            Default is `None`

            - `batch_size (B)` the number of samples per timestep
            - `n_units (HS)` the total number of OCM hidden neurons

        acm_h_state : jax.Array (optional)
            Hidden state for the ACM `(B*A, HS)`. If `None`, initializes to zeros.
            Default is `None`

            - `batch_size (B)` the number of samples per timestep
            - `n_actions (A)` the number of discrete actions
            - `n_units (HS)` the total number of ACM hidden neurons

        timespans : jax.Array (optional)
            Time intervals between observations `(T,)`. If `None`, uses uniform
            time intervals. Default is `None`

            - `seq_length (T)` the number of sequences (e.g., trajectories)

        Returns
        -------
        preds : AgentOutput
            An object of agent predictions
        h_state : AgentHiddenStates
            An object of the agents hidden states
        """

        # Process images
        encoded = self.cnn(obs)  # (B, T, F)

        # OCM forward - process observations and produce embeddings
        ocm_preds, ocm_h_state = self.ocm(
            encoded,
            h_state=ocm_h_state,
            timespans=timespans,
        )

        # ACM forward - use OCM embeddings for action-value predictions
        acm_preds, acm_h_state = self.acm(
            ocm_preds.embedding,
            h_state=acm_h_state,
            timespans=timespans,
        )

        return (
            AgentOutput.create(*ocm_preds.output_values(), *acm_preds.output_values()),
            AgentHiddenStates(ocm=ocm_h_state, acm=acm_h_state),
        )

    def act(self, logits: chex.Array) -> chex.Array:
        """
        Samples agent actions from the policy logits predicted by the agent.

        Parameters
        ----------
        logits : jax.Array
            Policy logits with shape `(B, A)` or `(B, T, A)`

        Returns
        -------
        actions : jax.Array
            Sampled actions `(B,)`
        """
        # Handle both squeezed (B, A) and non-squeezed (B, T, A) inputs
        if logits.ndim == 3:
            logits = jnp.squeeze(logits, axis=1)  # (B, 1, A) -> (B, A)

        # Reset RNG key for sampling
        self.key_actions, key_sample = jax.random.split(self.key_actions, 2)

        # Compute actions
        actions = distrax.Softmax(logits).sample(seed=key_sample)
        return actions
