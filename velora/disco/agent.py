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

import json
from pathlib import Path
from typing import Optional, Self, Tuple

import chex
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from velora.base.outputs import ParamCount
from velora.base.rollouts import Rollout
from velora.disco.config.settings import (
    DiscoAgentSettings,
    DiscoValueSettings,
    PolicyAgentSettings,
)
from velora.disco.config.state import PolicyAgentHiddenStates
from velora.disco.nn.encoder import DiscoInputEncoder, ImageEncoder
from velora.disco.nn.meta import DiscoNetwork
from velora.disco.nn.policy import ACM, OCM
from velora.disco.outputs import DiscoAgentOutput, PolicyAgentOutput
from velora.lnn.ncp import LNN
from velora.utils.format import create_directory
from velora.utils.nn import total_parameters
from velora.utils.seed import get_rng_key_data, restore_rng_key
from velora.utils.transforms import squeeze_time


class PolicyAgent:
    """
    Creates a policy agent used to discover Reinforcement Learning (RL) rules.

    Combines DiscoRL techniques with Liquid Neural Networks (LNNs).

    Architecture:
        1. Input Encoder - extracts visual features from image observations
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
    config : PolicyAgentSettings
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
        config: PolicyAgentSettings,
        key: chex.PRNGKey,
        jit_compile: bool = False,
    ) -> None:
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.config = config
        self.key = key

        self.n_actions = int(self.act_spec.n)

        key_encoder, key_ocm, key_acm, key_actions = jax.random.split(self.key, 4)
        self.key_actions = key_actions

        # Categorical bins for Q-values
        self.categorical_bins = config.categorical_bins()

        self._encoder = ImageEncoder(
            obs_spec.shape[-1],
            config.n_hidden,
            key=key_encoder,
        )

        self._ocm = OCM(
            self._encoder.output_dim,
            config.n_hidden,
            config.prediction_size,
            self.n_actions,
            key=key_ocm,
            sparsity=config.sparsity,
        )

        self._acm = ACM(
            self._ocm.embedding_size,
            config.n_hidden,
            config.prediction_size,
            self.n_actions,
            self.categorical_bins.num_bins,
            key=key_acm,
            sparsity=config.sparsity,
        )

        self.encoder, self.ocm, self.acm = self._compile(jit_compile)

    @property
    def active_params(self) -> int:
        """Get the agents active parameters."""
        return (
            self._encoder.active_params
            + self._ocm.active_params
            + self._acm.active_params
        )

    @property
    def total_params(self) -> int:
        """Get the agents total parameters."""
        return (
            self._encoder.total_params + self._ocm.total_params + self._acm.total_params
        )

    @property
    def param_count(self) -> ParamCount:
        """Get the agents parameter count."""
        return ParamCount(active=self.active_params, total=self.total_params)

    def _compile(self, jit_compile: bool) -> Tuple[ImageEncoder, OCM, ACM]:
        """
        Returns JIT-compiled or original modules based on compilation flag.

        Parameters
        ----------
        jit_compile : bool
            Whether to JIT compile the modules

        Returns
        -------
        encoder : ImageEncoder
            Input encoder (possibly JIT-wrapped)
        ocm : OCM
            Observation-Conditional Model (possibly JIT-wrapped)
        acm : ACM
            Action-Conditional Model (possibly JIT-wrapped)
        """
        if jit_compile:
            return nnx.jit(self._encoder), nnx.jit(self._ocm), nnx.jit(self._acm)  # type: ignore

        return self._encoder, self._ocm, self._acm

    def __call__(
        self,
        obs: chex.Array,
        *,
        ocm_h_state: Optional[chex.Array] = None,
        acm_h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[PolicyAgentOutput, PolicyAgentHiddenStates]:
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
        h_state : PolicyAgentHiddenStates
            An object of the agents hidden states
        """

        # Process images
        encoded = self.encoder(obs)  # (B, T, F)

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
            PolicyAgentOutput.create(
                *ocm_preds.output_values(),
                *acm_preds.output_values(),
            ),
            PolicyAgentHiddenStates(ocm=ocm_h_state, acm=acm_h_state),
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
            Sampled actions `(B, 1)`
        """
        # Handle both squeezed (B, A) and non-squeezed (B, T, A) inputs
        if logits.ndim == 3:
            logits = jnp.squeeze(logits, axis=1)  # (B, 1, A) -> (B, A)

        # Reset RNG key for sampling
        self.key_actions, key_sample = jax.random.split(self.key_actions, 2)

        # Compute actions
        actions = distrax.Softmax(logits).sample(seed=key_sample)
        return jnp.expand_dims(actions, axis=-1)

    def get_params(self) -> nnx.State:
        """
        Extract trainable parameters from all sub-modules.

        Returns
        -------
        params : nnx.State
            Combined parameter states from `(encoder, ocm, acm)`
        """
        return nnx.State(
            {
                "encoder": nnx.state(self._encoder, nnx.Param),
                "ocm": nnx.state(self._ocm, nnx.Param),
                "acm": nnx.state(self._acm, nnx.Param),
            }
        )

    def update_params(self, params: nnx.State) -> None:
        """
        Update parameters for all sub-modules.

        Parameters
        ----------
        params : nnx.State
            Parameter states to apply
        """
        nnx.update(self._encoder, params["encoder"])
        nnx.update(self._ocm, params["ocm"])
        nnx.update(self._acm, params["acm"])

    def soft_param_update(self, tau: float, new_params: nnx.State) -> None:
        """
        Performs a soft parameter update on the networks parameters.

        Formula: `θ_target ← τ * θ_online + (1 - τ) * θ_target`

        Parameters
        ----------
        tau : float
            Soft update coefficient
        new_params : nnx.State
            New parameters to use for updating (`θ_target`)
        """
        new_params = jax.tree.map(
            lambda old, new: tau * old + (1.0 - tau) * new,
            self.get_params(),
            new_params,
        )
        self.update_params(new_params)


class DiscoAgent:
    """
    Creates a target agent used to discover RL update (target) rules.

    Combines DiscoRL techniques with Liquid Neural Networks (LNNs).

    Architecture:
        - Input Encoder - converts buffer samples into embeddings for the Disco network
        - Disco Network - processes embeddings backwards through time to produce learned targets `(π̂, ŷ, ẑ)` for training the policy agent
        - Meta LNN - captures learning dynamics across the agent's lifetime, providing conditioning signals that modulate target generation
        - Meta Projection - projects meta conditioning to match encoder output to inject lifetime context into target generation

    Parameters
    ----------
    config : DiscoAgentSettings
        Configuration for the update rule agent
    key : jax.random.PRNGKey
        Random number generator key
    freeze : bool (optional)
        Freezes parameters so they cannot be trained.
        Useful for reusing trained target rules. Default is `False`
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    """

    def __init__(
        self,
        *,
        config: DiscoAgentSettings,
        key: chex.PRNGKey,
        freeze: bool = False,
        jit_compile: bool = False,
    ) -> None:
        self.config = config
        self.key = key
        self.is_frozen = freeze
        self.encoder_config = config.encoder_config()

        encoder_key, disco_key, meta_key, proj_key = jax.random.split(key, 4)

        self._encoder = DiscoInputEncoder(
            config=self.encoder_config,
            rngs=nnx.Rngs(encoder_key),
        )

        self._disco_net = DiscoNetwork(
            self.encoder_config.output_dim,
            self.config.n_hidden,
            self.config.prediction_size,
            self.encoder_config.action_embed_dim,
            key=disco_key,
            sparsity=self.config.sparsity,
        )

        self._meta_lnn = LNN(
            self.encoder_config.output_dim,
            self.config.n_hidden,
            self._disco_net.hidden_size,
            key=meta_key,
            sparsity=self.config.sparsity,
        )

        self._meta_proj = nnx.Linear(
            self._meta_lnn.hidden_size,
            self.encoder_config.output_dim,
            rngs=nnx.Rngs(proj_key),
        )

        self.encoder, self.disco_net, self.meta_lnn, self.meta_proj = self._compile(
            jit_compile
        )

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adan(self.config.lr),
        )

    @property
    def hidden_sizes(self) -> Tuple[int, int]:
        """
        Get the agents hidden sizes.

        Returns
        -------
        disco_h_size : int
            Disco network hidden size
        meta_h_size : int
            Meta LNN hidden size
        """
        return (self._disco_net.hidden_size, self._meta_lnn.hidden_size)

    @property
    def active_params(self) -> int:
        """Get the agents active parameters."""
        return (
            self._encoder.active_params
            + self._disco_net.active_params
            + self._meta_lnn.active_params
            + total_parameters(self._meta_proj)
        )

    @property
    def total_params(self) -> int:
        """Get the agents total parameters."""
        return (
            self._encoder.total_params
            + self._disco_net.total_params
            + self._meta_lnn.total_params
            + total_parameters(self._meta_proj)
        )

    @property
    def param_count(self) -> ParamCount:
        """Get the agents parameter count."""
        return ParamCount(active=self.active_params, total=self.total_params)

    def _compile(
        self, jit_compile: bool
    ) -> Tuple[DiscoInputEncoder, DiscoNetwork, LNN, nnx.Linear]:
        """
        Returns JIT-compiled or original modules based on compilation flag.

        Parameters
        ----------
        jit_compile : bool
            Whether to JIT compile the modules

        Returns
        -------
        encoder : DiscoInputEncoder
            Input encoder (possibly JIT-wrapped)
        disco_net : DiscoNetwork
            Disco network (possibly JIT-wrapped)
        meta_lnn : LNN
            Meta LNN (possibly JIT-wrapped)
        meta_proj : nnx.Linear
            Meta projection layer (possibly JIT-wrapped)
        """
        if jit_compile:
            return (
                nnx.jit(self._encoder),
                nnx.jit(self._disco_net),
                nnx.jit(self._meta_lnn),
                nnx.jit(self._meta_proj),
            )  # type: ignore

        return self._encoder, self._disco_net, self._meta_lnn, self._meta_proj

    def __call__(
        self,
        rollout: Rollout,
        disco_h_state: chex.Array | None = None,
        meta_h_state: chex.Array | None = None,
    ) -> Tuple[DiscoAgentOutput, chex.Array, chex.Array]:
        """
        Generate targets for agent training.

        Parameters
        ----------
        rollout : Rollout
            A trajectory of experience
        disco_h_state : chex.Array (optional)
            Hidden state for DiscoNetwork. Shape: `(B, H)`.
            Default is `None`
        meta_h_state : chex.Array (optional)
            Hidden state for MetaLNN. Shape: `(B, H_meta)`.
            Default is `None`

        Returns
        -------
        targets : DiscoAgentOutput
            Generated targets `(π̂, ŷ, ẑ)`
        disco_h_state : chex.Array
            Updated Disco network hidden state
        meta_h_state : chex.Array
            Updated Meta-LNN hidden state
        """
        # embed: (B, T, E), act_embed: (B, T, A, C)
        embedding, action_embed = self.encoder(rollout)
        disco_input = embedding

        # Apply meta conditioning from previous update (if available)
        if meta_h_state is not None:
            disco_input = self._meta_conditioning(embedding, meta_h_state)

        preds, disco_h_state = self.disco_net(
            disco_input,
            action_embed,
            h_state=disco_h_state,
        )

        # Update Meta-LNN for next iteration using trajectory summary
        summary = jnp.mean(embedding, axis=1)  # (B, T, E) -> (B, E)
        _, meta_h_state = self.meta_lnn(summary, h_state=meta_h_state)

        targets = DiscoAgentOutput(pi=preds.pi, y=preds.y, z=preds.z)

        # Freeze training
        if self.is_frozen:
            targets = jax.tree.map(jax.lax.stop_gradient, targets)
            disco_h_state = jax.lax.stop_gradient(disco_h_state)
            meta_h_state = jax.lax.stop_gradient(meta_h_state)

        return targets, disco_h_state, meta_h_state

    def _meta_conditioning(
        self, embedding: chex.Array, meta_h_state: chex.Array
    ) -> chex.Array:
        """
        Applies multiplicative interaction with meta conditioning to encoder output.

        Parameters
        ----------
        embedding : chex.Array
            Encoder output. Shape: `(B, T, E)`
        meta_h_state : chex.Array
            Meta-LNN hidden state. Shape: `(B, H_meta)`

        Returns
        -------
        result : chex.Array
            Conditioned embedding. Shape: `(B, T, E)`
        """
        # (B, H_meta) -> (B, E)
        new_h = self.meta_proj(meta_h_state)  # type: ignore
        new_h = jnp.expand_dims(new_h, axis=1)  # (B, E) -> (B, 1, E)
        return embedding * new_h

    def get_params(self) -> nnx.State:
        """
        Extract trainable parameters from all sub-modules.

        Returns
        -------
        params : nnx.State
            Combined parameter states from `(encoder, disco_net, meta_lnn, meta_proj)`
        """
        return nnx.State(
            {
                "encoder": nnx.state(self._encoder, nnx.Param),
                "disco_net": nnx.state(self._disco_net, nnx.Param),
                "meta_lnn": nnx.state(self._meta_lnn, nnx.Param),
                "meta_proj": nnx.state(self._meta_proj, nnx.Param),
            }
        )

    def update_params(self, params: nnx.State) -> None:
        """
        Update parameters for all sub-modules.

        Parameters
        ----------
        params : nnx.State
            Parameter states to apply
        """
        nnx.update(self._encoder, params["encoder"])
        nnx.update(self._disco_net, params["disco_net"])
        nnx.update(self._meta_lnn, params["meta_lnn"])
        nnx.update(self._meta_proj, params["meta_proj"])

    def save(
        self,
        root_path: Path | str = "checkpoints/models",
        model_dir: str = "disco",
        timestamp: bool = True,
    ) -> Path:
        """
        Saves the agent's network parameters and configuration to disk.

        Useful for saving discovered update rules.

        Default directory path: `./checkpoints/models/disco_[ddmmyy]_[hhmmss]/`.

        Parameters
        ----------
        root_path : Path | str (optional)
            Directory path for saving. Default is `checkpoints/models`
        model_dir : str (optional)
            The folder name to save the agents state. Gets merged with `root_path`.
            Default is `disco`
        timestamp : bool (optional)
            Whether to append timestamps to experiment directory.
            Uses timestamp format: `ddmmyy_hhmmss`. Default is `True`

        Returns
        -------
        path : Path
            Path where the rule was saved
        """
        dirpath = create_directory(root_path, model_dir, timestamp)
        dirpath.mkdir(parents=True, exist_ok=True)

        cp = ocp.StandardCheckpointer()

        # Set config as JSON
        key_data = get_rng_key_data(self.key)
        config_json = self.config.to_json(key=key_data)

        # Save params and config
        cp.save(dirpath / "params", self.get_params())
        (dirpath / "config.json").write_text(config_json)

        cp.wait_until_finished()
        return dirpath

    @classmethod
    def load(
        cls,
        path: Path | str,
        freeze: bool = True,
        jit_compile: bool = False,
    ) -> Self:
        """
        Restore a saved agents state. Useful for loading a discovered update rule.

        Parameters
        ----------
        path : Path | str
            Directory path to load from
        freeze : bool (optional)
            Freezes parameters so they cannot be trained. Default is `True`
        jit_compile : bool (optional)
            Whether to JIT compile. Default is `False`

        Returns
        -------
        disco_agent : DiscoAgent
            Agent with restored parameters
        """
        path = Path(path).resolve()

        # Load config
        config_dict: dict = json.loads((path / "config.json").read_text())
        key = restore_rng_key(config_dict.pop("key"))
        config = DiscoAgentSettings(**config_dict)

        agent = cls(
            config=config,
            key=key,
            freeze=freeze,
            jit_compile=jit_compile,
        )

        # Restore parameters
        abstract_params = jax.tree.map(
            ocp.utils.to_shape_dtype_struct,
            agent.get_params(),
        )

        cp = ocp.StandardCheckpointer()
        params = cp.restore(path / "params", abstract_params)

        agent.update_params(params)
        return agent


class DiscoValueAgent:
    """
    Value function agent for computing state values `V(s)` during meta-training.

    Estimates advantages for computing meta-gradients during target rule
    discovery.

    Architecture:
        1. Input Encoder - extracts visual features from image observations
        2. LNN - processes encoded features to produce
           state value estimates

    Parameters
    ----------
    obs_spec : gym.spaces.Box
        A single observation space of the vectorized Gymnasium environment
    n_hidden : int
        Number of decision nodes for policy networks (inter + command nodes)
    config : DiscoValueSettings
        Value function configuration settings
    key : jax.random.PRNGKey
        Random number generator key
    sparsity : float (optional)
        Network connection sparsity between neurons used for the Liquid Neural
        Networks (LNNs). Default is `0.5`.

        Must be a value between `[0.1, 0.9]`:

            - Where `0.1` neurons are very dense
            - Where `0.9` neurons are very sparse
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    """

    def __init__(
        self,
        obs_spec: gym.spaces.Box,
        n_hidden: int,
        *,
        config: DiscoValueSettings,
        key: chex.PRNGKey,
        sparsity: float = 0.5,
        jit_compile: bool = False,
    ) -> None:
        self.obs_spec = obs_spec
        self.config = config
        self.key = key

        key_encoder, key_value = jax.random.split(self.key, 2)

        self._encoder = ImageEncoder(
            obs_spec.shape[-1],
            n_hidden,
            key=key_encoder,
        )

        self._net = LNN(
            self._encoder.output_dim,
            n_hidden,
            out_features=1,
            key=key_value,
            sparsity=sparsity,
        )

        self.encoder, self.net = self._compile(jit_compile)

    @property
    def active_params(self) -> int:
        """Get the agents active parameters."""
        return self._encoder.active_params + self._net.active_params

    @property
    def total_params(self) -> int:
        """Get the agents total parameters."""
        return self._encoder.total_params + self._net.total_params

    @property
    def param_count(self) -> ParamCount:
        """Get the agents parameter count."""
        return ParamCount(active=self.active_params, total=self.total_params)

    def _compile(self, jit_compile: bool) -> Tuple[ImageEncoder, LNN]:
        """
        Returns JIT-compiled or original modules based on compilation flag.

        Parameters
        ----------
        jit_compile : bool
            Whether to JIT compile the modules

        Returns
        -------
        encoder : ImageEncoder
            Input encoder (possibly JIT-wrapped)
        net : LNN
            Value network (possibly JIT-wrapped)
        """
        if jit_compile:
            return nnx.jit(self._encoder), nnx.jit(self._net)  # type: ignore

        return self._encoder, self._net

    def __call__(
        self,
        obs: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array]:
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

        h_state : jax.Array (optional)
            Hidden state for the network `(B, HS)`. If `None`, initializes to zeros.
            Default is `None`

            - `batch_size (B)` the number of samples per timestep
            - `n_units (HS)` the total number of hidden neurons (`n_hidden + 1`)

        timespans : jax.Array (optional)
            Time intervals between observations `(T,)`. If `None`, uses uniform
            time intervals. Default is `None`

            - `seq_length (T)` the number of sequences (e.g., trajectories)

        Returns
        -------
        v : chex.Array
            State-value estimates with shape `(B, T, 1)` or `(B, 1)` if `T=1`
        h_state : chex.Array
            Updated hidden state. Shape: `(B, HS)`
        """
        # Encode images -> (B, T, F)
        features = self.encoder(obs)

        # Compute state-value -> (B, T, 1)
        v, h_state = self.net(
            features,
            h_state=h_state,
            timespans=timespans,
        )

        # (B, T, 1) -> (B, 1) if T=1
        v = squeeze_time(v)
        return v, h_state

    def get_params(self) -> nnx.State:
        """
        Extract trainable parameters from all sub-modules.

        Returns
        -------
        params : nnx.State
            Combined parameter states from `(encoder, net)`
        """
        return nnx.State(
            {
                "encoder": nnx.state(self._encoder, nnx.Param),
                "net": nnx.state(self._net, nnx.Param),
            }
        )

    def update_params(self, params: nnx.State) -> None:
        """
        Update parameters for all sub-modules.

        Parameters
        ----------
        params : nnx.State
            Parameter states to apply
        """
        nnx.update(self._encoder, params["encoder"])
        nnx.update(self._net, params["net"])
