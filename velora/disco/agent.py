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

from velora.disco.config.settings import DiscoAgentSettings, PolicyAgentSettings
from velora.disco.nn.decoder import GaussianPolicyDecoder
from velora.disco.nn.encoder import DiscoInputEncoder
from velora.disco.nn.meta import DiscoNetwork
from velora.disco.nn.policy import ACM, OCM
from velora.disco.outputs import DiscoAgentOutput, PolicyAgentOutput
from velora.disco.rollouts import Rollout
from velora.lnn.ncp import LNN


class PolicyAgent(nnx.Module):
    """
    Creates a policy agent used to discover Reinforcement Learning (RL) rules.

    Combines DiscoRL target-based learning with Liquid Neural Networks (LNNs)
    and Gaussian policy parameterization. Suitable for continuous action spaces.

    Architecture:
        1. Observation-Conditional Model (OCM) - processes encoded features
           to produce policy hidden representation and observation-conditioned
           predictions `(y)`
        2. Policy Head - projects OCM pi_hidden to Gaussian parameters
           `μ(s)` and `log σ(s)` at `max_action_dim` width
        3. Continuous Action-Conditional Model (ACM) - takes OCM
           embeddings concatenated with the continuous action taken, producing
           action-conditioned predictions `z(s, a)`, auxiliary policy
           predictions `aux_pi(s, a)`, and scalar Q-values `q(s, a)`
        4. Auxiliary Policy Head - projects ACM aux_pi hidden to predicted
           next-step Gaussian parameters `(μ', log σ')` at `2 * max_action_dim`

    Output Shapes:
        - μ (policy mean): `(B, T, D)` padded to `max_action_dim`
        - log_σ (policy log-std): `(B, T, D)` padded to `max_action_dim`
        - y (obs-conditioned prediction): `(B, T, Y)`
        - z (action-conditioned prediction): `(B, T, Z)` — no action dimension
        - aux_pi (auxiliary policy prediction): `(B, T, 2D)` — predicted `(μ', log σ')`
        - q (action-value): `(B, T, 1)` — scalar

    References:
        - [Liquid Time-constant Networks (2020)](https://arxiv.org/abs/2006.04439)

    Parameters
    ----------
    encoding_dim : int
        OCM encoding input dimension
    max_action_dim : int
        Maximum continuous action dimensionality across all environments
        in the training set. Actions are zero-padded to this size
    config : PolicyAgentSettings
        Configuration for the policy agent
    key : jax.random.PRNGKey
        Random number generator key
    """

    def __init__(
        self,
        encoding_dim: int,
        max_action_dim: int,
        *,
        config: PolicyAgentSettings,
        key: chex.PRNGKey,
    ) -> None:
        key_ocm, key_acm, key_decoder = jax.random.split(key, 3)

        self.ocm = OCM(
            encoding_dim,
            config.n_hidden,
            config.prediction_size,
            key=key_ocm,
            sparsity=config.sparsity,
        )

        # ACM: takes (ocm_embedding, action) -> (z, aux_pi, q)
        self.acm = ACM(
            self.ocm.embedding_size,
            max_action_dim,
            config.n_hidden,
            config.prediction_size,
            key=key_acm,
            sparsity=config.sparsity,
        )

        # Policy decoder
        self.decoder = GaussianPolicyDecoder(
            config.prediction_size,
            max_action_dim,
            min_log_std=config.min_log_std,
            max_log_std=config.max_log_std,
            key=key_decoder,
        )

    def __call__(
        self,
        encoding: jax.Array,
        action: jax.Array,
        *,
        action_dim_mask: jax.Array,
        ocm_h: jax.Array | None = None,
        acm_h: jax.Array | None = None,
    ) -> Tuple[PolicyAgentOutput, jax.Array, jax.Array]:
        """
        Forward pass through the module.

        Parameters
        ----------
        encoding : jax.Array
           Encoder embeddings `(B, T, F)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories)
            - `features (F)` number of features in the embedding
        action : jax.Array
            Continuous action vector `(B, T, D)` padded to `max_action_dim`
        action_dim_mask : jax.Array
            Boolean mask `(max_action_dim,)` for valid action dimensions
        ocm_h : jax.Array (optional)
            Current OCM hidden state. Default is `None`
        acm_h : jax.Array (optional)
            Current ACM hidden state. Default is `None`

        Returns
        -------
        preds : PolicyAgentOutput
            Agent predictions
        ocm_h : jax.Array
            Updated OCM hidden state
        acm_h : jax.Array
            Updated ACM hidden state
        """
        ocm_preds, ocm_h = self.ocm(encoding, h_state=ocm_h)
        acm_preds, acm_h = self.acm(ocm_preds.embedding, action, h_state=acm_h)

        mu, log_std, aux_pi = self.decoder(
            ocm_preds.pi,
            acm_preds.aux_pi,
            action_dim_mask=action_dim_mask,
        )

        output = PolicyAgentOutput.create(
            encoding,
            mu,
            log_std,
            ocm_preds.y,
            acm_preds.z,
            aux_pi,
            acm_preds.q,
        )
        return output, ocm_h, acm_h


class DiscoAgent(nnx.Module):
    """
    Creates a target agent used to discover RL update (target) rules.

    Combines DiscoRL techniques with Liquid Neural Networks (LNNs).
    Suitable for continuous action spaces.

    Architecture:
        - Input Encoder - converts samples into embeddings for the Disco network
        - Disco Network - processes embeddings backwards through time to produce
          learned targets `(μ̂, log σ̂, ŷ, ẑ)` for training the policy agent
        - Meta LNN - captures learning dynamics across the agent's lifetime,
          providing conditioning signals that modulate target generation
        - Meta Projection - projects meta conditioning to match encoder output
          to inject lifetime context into target generation

    Parameters
    ----------
    max_action_dim : int
        Maximum continuous action dimensionality across all environments.
        Used to size the policy target projections and encoder inputs.
    config : DiscoAgentSettings
        Configuration for the update rule agent
    key : jax.random.PRNGKey
        Random number generator key
    freeze : bool (optional)
        Freezes parameters so they cannot be trained.
        Useful for reusing trained target rules. Default is `False`
    """

    def __init__(
        self,
        max_action_dim: int,
        *,
        config: DiscoAgentSettings,
        key: chex.PRNGKey,
        freeze: bool = False,
    ) -> None:
        self.is_frozen = freeze

        encoder_config = config.encoder_config()
        encoder_key, disco_key, meta_key, proj_key = jax.random.split(key, 4)

        self.encoder = DiscoInputEncoder(
            encoder_config,
            max_action_dim,
            rngs=nnx.Rngs(encoder_key),
        )

        self.disco_net = DiscoNetwork(
            encoder_config.output_dim,
            config.n_hidden,
            config.prediction_size,
            max_action_dim,
            key=disco_key,
            sparsity=config.sparsity,
        )

        self.meta_lnn = LNN(
            encoder_config.output_dim,
            config.n_hidden,
            self.disco_net.hidden_size,
            key=meta_key,
            sparsity=config.sparsity,
        )

        self.meta_proj = nnx.Linear(
            self.meta_lnn.hidden_size,
            encoder_config.output_dim,
            rngs=nnx.Rngs(proj_key),
        )

    @property
    def hidden_sizes(self) -> Tuple[int, int]:
        """
        Get the agent's hidden sizes.

        Returns
        -------
        disco_h_size : int
            Disco network hidden size
        meta_h_size : int
            Meta LNN hidden size
        """
        return (self.disco_net.hidden_size, self.meta_lnn.hidden_size)

    def __call__(
        self,
        rollout: Rollout,
        *,
        disco_h_state: jax.Array | None = None,
        meta_h_state: jax.Array | None = None,
    ) -> Tuple[DiscoAgentOutput, jax.Array, jax.Array]:
        """
        Generate targets for agent training.

        Parameters
        ----------
        rollout : Rollout
            A trajectory of experience
        disco_h_state : jax.Array (optional)
            Hidden state for DiscoNetwork. Shape: `(B, H)`.
            Default is `None`
        meta_h_state : jax.Array (optional)
            Hidden state for MetaLNN. Shape: `(B, H_meta)`.
            Default is `None`

        Returns
        -------
        targets : DiscoAgentOutput
            Generated targets `(μ̂, log σ̂, ŷ, ẑ)`
        disco_h_state : jax.Array
            Updated Disco network hidden state
        meta_h_state : jax.Array
            Updated Meta-LNN hidden state
        """
        # embed: (B, T, E)
        embedding = self.encoder(rollout)
        disco_input = embedding

        # Apply meta conditioning from previous update (if available)
        if meta_h_state is not None:
            new_h = self.meta_proj(meta_h_state)  # (B, H_meta) -> (B, E)
            new_h = jnp.expand_dims(new_h, axis=1)  # (B, E) -> (B, 1, E)
            disco_input = embedding * new_h

        preds, disco_h_state = self.disco_net(disco_input, h_state=disco_h_state)

        # Update Meta-LNN for next iteration using trajectory summary
        summary = jnp.mean(embedding, axis=1)  # (B, T, E) -> (B, E)
        _, meta_h_state = self.meta_lnn(summary, h_state=meta_h_state)

        targets = DiscoAgentOutput(
            mu=preds.mu,
            log_std=preds.log_std,
            y=preds.y,
            z=preds.z,
        )

        # Freeze training
        if self.is_frozen:
            targets = jax.tree.map(jax.lax.stop_gradient, targets)
            disco_h_state = jax.lax.stop_gradient(disco_h_state)
            meta_h_state = jax.lax.stop_gradient(meta_h_state)

        return targets, disco_h_state, meta_h_state
