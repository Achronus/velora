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

from abc import abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Generic,
    NamedTuple,
    Optional,
    Self,
    Tuple,
    TypeVar,
    get_args,
)

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from velora.base.outputs import ParamCount
from velora.disco.config.metadata import RuleTrainerMetadata
from velora.disco.config.settings import (
    DiscoAgentSettings,
    DiscoValueSettings,
    PolicyAgentSettings,
)
from velora.disco.config.state import PolicyAgentHiddenStates, RuleTrainerState
from velora.disco.nn.decoder import ActionDecoder, GaussianPolicyDecoder
from velora.disco.nn.encoder import (
    ContinuousDiscoInputEncoder,
    DiscoInputEncoder,
    ImageEncoder,
    PolicyEncoder,
    build_obs_encoder,
)
from velora.disco.nn.meta import ContinuousDiscoNetwork, DiscoNetwork
from velora.disco.nn.policy import ACM, OCM, ContinuousACM
from velora.disco.outputs import (
    ACMPredictions,
    ContinuousDiscoAgentOutput,
    ContinuousPolicyAgentOutput,
    DiscoAgentOutput,
    OCMPredictions,
    PolicyAgentOutput,
)
from velora.disco.rollouts import ContinuousRollout, Rollout
from velora.lnn.base import BaseCfC
from velora.lnn.ncp import LNN
from velora.tracking.manager import CheckpointManager
from velora.tracking.settings import RunSettings
from velora.utils.format import create_directory
from velora.utils.nn import total_parameters
from velora.utils.seed import get_rng_key_data, restore_rng_key
from velora.utils.transforms import squeeze_time


class DiscoModules(NamedTuple):
    encoder: DiscoInputEncoder
    disco_net: DiscoNetwork
    meta_lnn: LNN
    meta_proj: nnx.Linear


class ContinuousDiscoModules(NamedTuple):
    encoder: ContinuousDiscoInputEncoder
    disco_net: ContinuousDiscoNetwork
    meta_lnn: LNN
    meta_proj: nnx.Linear


class PolicyModules(NamedTuple):
    encoder: ImageEncoder
    ocm: OCM
    acm: ACM
    decoder: ActionDecoder


class ContinuousPolicyModules(NamedTuple):
    encoder: PolicyEncoder
    ocm: OCM
    acm: ContinuousACM
    decoder: GaussianPolicyDecoder


class ValueModules(NamedTuple):
    encoder: ImageEncoder
    net: LNN


ModulesT = TypeVar("ModulesT", bound=tuple)


class BaseAgent(Generic[ModulesT]):
    """
    Base class for all agents. Auto-discovers sub-modules and provides
    shared parameter management, compilation, and merge logic.

    Sub-modules are auto-detected by scanning `self.__dict__` for private
    attributes (`_[name]`) that have an `active_params` property — these are
    neural network modules. Detection happens during `_compile()`.

    Provides
    --------
    - `active_params`, `total_params`, `param_count` — aggregate counts
    - `get_params()`, `update_params()` — extract / apply parameter states
    - `merge_params()` — reconstruct modules from explicit parameters
    - `_compile()` — cache graphdefs, optionally JIT-compile all modules
    """

    _module_registry: Dict[str, str]
    _modules_type: type

    def _discover_modules(self) -> Dict[str, str]:
        """
        Scan `self.__dict__` for private attributes that are NN modules.

        A module is identified by having an `active_params` attribute.

        Returns
        -------
        registry : Dict[str, str]
            Mapping of public name to private attribute name,
            e.g., `{"encoder": "_encoder", "disco_net": "_disco_net"}`
        """
        registry = {}
        for attr_name, obj in self.__dict__.items():
            if attr_name.startswith("_") and isinstance(obj, nnx.Module):
                public_name = attr_name.lstrip("_")
                registry[public_name] = attr_name

        return registry

    def _resolve_modules_type(self) -> type:
        """
        Resolve the concrete `NamedTuple` type bound to `ModulesT`
        by walking the class MRO.

        Returns
        -------
        modules_type : type
            The concrete NamedTuple class (e.g., `DiscoModules`)
        """
        for cls in type(self).__mro__:
            for base in getattr(cls, "__orig_bases__", ()):
                for arg in get_args(base):
                    if isinstance(arg, type) and issubclass(arg, tuple):
                        return arg

        raise TypeError(
            f"{type(self).__name__} must bind ModulesT to a NamedTuple "
            f"(e.g., class MyAgent(BaseAgent[MyModules]))"
        )

    def _get_param_count(self, counter: str) -> int:
        """
        Sum a parameter count across all registered modules.

        Uses the module's own `counter` property (e.g., `active_params`,
        `total_params`) when available. Falls back to `total_parameters()`
        for plain `nnx.Module` instances that don't track counts.

        Parameters
        ----------
        counter : str
            Name of the property to read from each module

        Returns
        -------
        count : int
            Aggregate parameter count
        """
        total = 0
        for attr in self._module_registry.values():
            module = getattr(self, attr)

            if hasattr(module, counter):
                total += getattr(module, counter)
            else:
                total += total_parameters(module)

        return total

    @property
    def active_params(self) -> int:
        """Get the agent's active parameter count."""
        return self._get_param_count("active_params")

    @property
    def total_params(self) -> int:
        """Get the agent's total parameter count."""
        return self._get_param_count("total_params")

    @property
    def param_count(self) -> ParamCount:
        """Get the agent's parameter count."""
        return ParamCount(active=self.active_params, total=self.total_params)

    def get_params(self) -> nnx.State:
        """
        Extract trainable parameters from all sub-modules.

        Returns
        -------
        params : nnx.State
            Combined parameter states from all registered modules
        """
        return nnx.State(
            {
                name: nnx.state(getattr(self, attr), nnx.Param)
                for name, attr in self._module_registry.items()
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
        for name, attr in self._module_registry.items():
            nnx.update(getattr(self, attr), params[name])

    def merge_params(self, params: nnx.State) -> ModulesT:
        """
        Reconstruct all modules from cached graphdefs and explicit parameters.

        Parameters
        ----------
        params : nnx.State
            Module parameters from `get_params()`

        Returns
        -------
        modules : ModulesT
            Reconstructed modules in registration order
        """
        return self._modules_type._make(  # type: ignore[attr-defined]
            nnx.merge(
                getattr(self, f"_{name}_graphdef"),
                params[name],
                getattr(self, f"_{name}_rest"),
            )
            for name in self._module_registry
        )

    def _compile(self, jit_compile: bool) -> ModulesT:
        """
        Auto-discover modules, cache graphdefs, and optionally JIT-compile.

        Must be called at the end of subclass `__init__` after all
        `self._xxx` module attributes have been set.

        Parameters
        ----------
        jit_compile : bool
            Whether to JIT compile the modules

        Returns
        -------
        modules : ModulesT
            Compiled (or original) modules in registration order
        """
        self._module_registry = self._discover_modules()
        self._modules_type = self._resolve_modules_type()

        for name, attr in self._module_registry.items():
            module = getattr(self, attr)
            graphdef, _, rest = nnx.split(module, nnx.Param, ...)
            setattr(self, f"_{name}_graphdef", graphdef)
            setattr(self, f"_{name}_rest", rest)

        modules = [getattr(self, attr) for attr in self._module_registry.values()]

        if jit_compile:
            modules = [nnx.jit(m) for m in modules]

        return self._modules_type._make(modules)  # type: ignore[attr-defined]


class BaseDiscoAgent(BaseAgent[ModulesT]):
    """
    Base class for disco target agents.
    """

    config: DiscoAgentSettings
    key: chex.PRNGKey
    is_frozen: bool

    _encoder: nnx.Module
    _disco_net: BaseCfC

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

        # Subclass sets self._encoder and self._disco_net
        self._build_modules(encoder_key, disco_key)

        # Shared across both variants
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

    @abstractmethod
    def _build_modules(
        self,
        encoder_key: chex.PRNGKey,
        disco_key: chex.PRNGKey,
    ) -> None:
        """
        Subclass hook to build encoder and disco network.

        Must set `self._encoder` and `self._disco_net`.

        Parameters
        ----------
        encoder_key : chex.PRNGKey
            RNG key for encoder initialization
        disco_key : chex.PRNGKey
            RNG key for disco network initialization
        """
        raise NotImplementedError

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
        return (self._disco_net.hidden_size, self._meta_lnn.hidden_size)

    def _meta_conditioning(
        self,
        meta_proj: nnx.Linear,
        embedding: chex.Array,
        meta_h_state: chex.Array,
    ) -> chex.Array:
        """
        Applies multiplicative interaction with meta conditioning to encoder output.

        Parameters
        ----------
        meta_proj : nnx.Linear
            The active meta projection layer
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
        new_h = meta_proj(meta_h_state)  # type: ignore
        new_h = jnp.expand_dims(new_h, axis=1)  # (B, E) -> (B, 1, E)
        return embedding * new_h

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
        run_dir: Path | str,
        *,
        checkpoint_step: int | None = None,
        freeze: bool = True,
        jit_compile: bool = False,
    ) -> Self:
        """
        Load a discovered update rule from a run directory.

        Reads agent configuration and key from `metadata.json`,
        then restores parameters from the specified (or latest) step
        checkpoint.

        Parameters
        ----------
        run_dir : Path | str
            Path to the run directory containing `metadata.json`
            and `checkpoints/`
        checkpoint_step : int (optional)
            Specific checkpoint step to restore. When `None`,
            restores the latest. Default is `None`
        freeze : bool (optional)
            Freezes parameters so they cannot be trained. Default is `True`
        jit_compile : bool (optional)
            Whether to JIT compile. Default is `False`

        Returns
        -------
        agent : Self
            Agent with restored parameters
        """
        run_settings = RunSettings.from_path(run_dir)

        with CheckpointManager(run_settings) as manager:
            meta = manager.load_metadata(RuleTrainerMetadata)

        config = DiscoAgentSettings(**meta.config["disco_agent"])
        key = restore_rng_key(meta.disco_key)

        agent = cls(
            config=config,
            key=key,
            freeze=freeze,
            jit_compile=jit_compile,
        )

        # Build full checkpoint template
        params = agent.get_params()
        meta_lr = meta.config.get("meta_lr", 0.001)
        num_trainers = len(meta.envs) * meta.agents_per_env
        batch_size = meta.config.get("batch_size", 20)

        dummy_state = RuleTrainerState.create(
            optax.adam(meta_lr).init(params),
            num_trainers,
            batch_size,
            *agent.hidden_sizes,
        )

        abstract_checkpoint = {
            "meta_params": jax.tree.map(
                ocp.utils.to_shape_dtype_struct,
                params,
            ),
            "state": jax.tree.map(
                lambda x: ocp.utils.to_shape_dtype_struct(x) if x is not None else None,
                dummy_state,
            ),
        }

        with CheckpointManager(run_settings) as manager:
            restored = manager.restore(checkpoint_step, abstract_checkpoint)

        agent.update_params(restored["meta_params"])
        return agent


class BasePolicyAgent(BaseAgent[ModulesT]):
    """
    Base class for policy agents.
    """

    config: PolicyAgentSettings
    key: chex.PRNGKey

    _encoder: PolicyEncoder

    @property
    def encoding_dim(self) -> int:
        """Output feature dimensionality of the encoder."""
        return self._encoder.encoding_dim

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


class PolicyAgent(BasePolicyAgent[PolicyModules]):
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
        4. Action Decoder - decodes concatenated OCM and ACM hidden
           representations into per-action outputs

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
    max_actions : int
        Maximum number of discrete actions across all environments in the
        training set
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
        max_actions: int,
        jit_compile: bool = False,
    ) -> None:
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.config = config
        self.key = key

        self.n_actions = int(self.act_spec.n)
        self.max_actions = max_actions

        key_encoder, key_ocm, key_acm, key_decoder, key_actions = jax.random.split(
            self.key,
            5,
        )
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
            key=key_ocm,
            sparsity=config.sparsity,
        )

        self._acm = ACM(
            self._ocm.embedding_size,
            config.n_hidden,
            config.prediction_size,
            self.categorical_bins.num_bins,
            key=key_acm,
            sparsity=config.sparsity,
        )

        self._decoder = ActionDecoder(
            in_dim=config.prediction_size * 3 + self.categorical_bins.num_bins,
            max_actions=self.max_actions,
            pi_dim=1,
            z_dim=config.prediction_size,
            aux_pi_dim=self.max_actions,
            q_dim=self.categorical_bins.num_bins,
            key=key_decoder,
        )

        self.encoder, self.ocm, self.acm, self.decoder = self._compile(jit_compile)

    def __call__(
        self,
        obs: chex.Array,
        *,
        ocm_h_state: Optional[chex.Array] = None,
        acm_h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
        encoding: Optional[chex.Array] = None,
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
            Hidden state for the ACM `(B, HS)`. If `None`, initializes to zeros.
            Default is `None`

            - `batch_size (B)` the number of samples per timestep
            - `n_units (HS)` the total number of ACM hidden neurons

        timespans : jax.Array (optional)
            Time intervals between observations `(T,)`. If `None`, uses uniform
            time intervals. Default is `None`

            - `seq_length (T)` the number of sequences (e.g., trajectories)

        encoding : jax.Array (optional)
            Pre-computed encoder features `(B, T, F)`. When provided,
            skips the encoder forward pass. Default is `None`

        Returns
        -------
        preds : AgentOutput
            An object of agent predictions
        h_state : PolicyAgentHiddenStates
            An object of the agents hidden states
        """
        # Reuse pre-computed encoding or compute fresh
        if encoding is None:
            encoding = self.encoder(obs)  # (B, T, F)

        # OCM forward - process observations and produce embeddings
        ocm_preds, ocm_h_state = self.ocm(
            encoding,
            h_state=ocm_h_state,
            timespans=timespans,
        )

        # ACM forward - use OCM embeddings for action-value predictions
        acm_preds, acm_h_state = self.acm(
            ocm_preds.embedding,
            h_state=acm_h_state,
            timespans=timespans,
        )

        # Decode to per-action outputs at max_actions
        preds = self._decode_to_actions(encoding, ocm_preds, acm_preds)

        return (
            preds,
            PolicyAgentHiddenStates(ocm=ocm_h_state, acm=acm_h_state),
        )

    def _decode_to_actions(
        self,
        encoding: chex.Array,
        ocm_preds: OCMPredictions,
        acm_preds: ACMPredictions,
        *,
        action_mask: chex.Array | None = None,
        decoder: ActionDecoder | None = None,
    ) -> PolicyAgentOutput:
        """
        Decode fixed-size hidden representations to per-action outputs.

        Always decodes at `max_actions` for identically-shaped outputs
        regardless of environment action count. Invalid action slots
        are masked to `-1e9`. Using Softmax will assign them to
        zero-probability.

        Parameters
        ----------
        encoding : chex.Array
            Encoder embeddings `(B, T, F)`
        ocm_preds : OCMPredictions
            OCM output with fixed-size `pi` and `y` heads
        acm_preds : ACMPredictions
            ACM output with fixed-size `z`, `aux_pi`, `q` heads
        action_mask : chex.Array (optional)
            Boolean mask `(max_actions,)` for valid actions.
            Default is `None`
        decoder : ActionDecoder (optional)
        Reconstructed decoder from explicit params for the functional
        gradient path. Uses `self.decoder` when `None`.
        Default is `None`

        Returns
        -------
        preds : PolicyAgentOutput
            Per-action agent predictions at `max_actions` dimension
        """
        d = decoder if decoder is not None else self.decoder

        heads = d(
            ocm_preds.pi,
            acm_preds.z,
            acm_preds.aux_pi,
            acm_preds.q,
        )

        pi = jnp.squeeze(heads.pi, axis=-1)  # (B, T, max_A, 1) -> (B, T, max_A)
        aux_pi = heads.aux_pi  # (B, T, max_A, max_A)

        # Mask padded action slots so softmax assigns zero probability
        if action_mask is None:
            action_mask = jnp.arange(self.max_actions) < self.n_actions  # (max_A,)

        pi = jnp.where(action_mask, pi, -1e9)  # (B, T, max_A)
        aux_pi = jnp.where(action_mask, aux_pi, -1e9)  # (B, T, max_A, max_A)

        return PolicyAgentOutput.create(
            encoding,
            pi,
            ocm_preds.y,
            heads.z,
            aux_pi,
            heads.q,
        )

    def functional_forward(
        self,
        encoding: chex.Array,
        params: nnx.State,
        *,
        action_mask: chex.Array | None = None,
    ) -> PolicyAgentOutput:
        """
        Pure functional forward through OCM + ACM + ActionDecoder
        with explicit params. Safe for use inside `jax.grad`.

        Always decodes at `max_actions`. Padded action slots are masked
        to `-1e9` using the provided `action_mask` (vmap path) or
        `self.n_actions` (standalone).

        Parameters
        ----------
        encoding : chex.Array
           Encoder embeddings `(B, T, F)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories)
            - `features (F)` number of features in the embedding
        params : nnx.State
            OCM, ACM and decoder parameters
        action_mask : chex.Array (optional)
            Boolean mask `(max_actions,)` for valid actions. Required in
            the vmap path where `self.n_actions` reflects the template
            trainer, not the actual trainer. Default is `None`

        Returns
        -------
        preds : AgentOutput
            Agent predictions at `max_actions` dimension
        """
        _, ocm, acm, decoder = self.merge_params(params)

        ocm_preds, _ = ocm(encoding)
        acm_preds, _ = acm(ocm_preds.embedding)

        return self._decode_to_actions(
            encoding,
            ocm_preds,
            acm_preds,
            action_mask=action_mask,
            decoder=decoder,
        )

    def act(self, logits: chex.Array) -> np.ndarray:
        """
        Samples agent actions from the policy logits predicted by the agent.

        Parameters
        ----------
        logits : chex.Array
            Policy logits with shape `(B, A)` or `(B, T, A)`

        Returns
        -------
        actions : np.Array
            Sampled actions `(B, 1)`
        """
        logits = np.asarray(logits)

        # Handle both squeezed (B, A) and non-squeezed (B, T, A) inputs
        if logits.ndim == 3:
            logits = logits.squeeze(axis=1)  # (B, 1, A) -> (B, A)

        def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
            e: np.ndarray = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e / e.sum(axis=axis, keepdims=True)

        # Compute actions
        probs = softmax(logits, axis=-1)
        cumprobs = np.cumsum(probs, axis=-1)
        cumprobs[:, -1] = 1.0  # Guarantee last bucket catches everything
        u = np.random.uniform(size=(probs.shape[0], 1))
        actions: np.ndarray = (u < cumprobs).argmax(axis=-1)[:, None]

        return actions.astype(np.int32)


class DiscoAgent(BaseDiscoAgent[DiscoModules]):
    """
    Creates a target agent used to discover RL update (target) rules.

    Combines DiscoRL techniques with Liquid Neural Networks (LNNs).

    Architecture:
        - Input Encoder - converts samples into embeddings for the Disco network
        - Disco Network - processes embeddings backwards through time to produce
          learned targets `(π̂, ŷ, ẑ)` for training the policy agent
        - Meta LNN - captures learning dynamics across the agent's lifetime,
          providing conditioning signals that modulate target generation
        - Meta Projection - projects meta conditioning to match encoder output
          to inject lifetime context into target generation

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

    def _build_modules(
        self, encoder_key: chex.PRNGKey, disco_key: chex.PRNGKey
    ) -> None:
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

    def __call__(
        self,
        rollout: Rollout,
        *,
        disco_h_state: chex.Array | None = None,
        meta_h_state: chex.Array | None = None,
        params: nnx.State | None = None,
        action_mask: chex.Array | None = None,
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
        params : nnx.State (optional)
            Meta-agent parameters from `get_params()`. Only required if
            using a functional approach for `jax.grad`. Default is `None`
        action_mask : chex.Array (optional)
            Boolean mask `(max_actions,)` for valid actions. Passed to encoder
            for masked mean over action embeddings. Default is `None`

        Returns
        -------
        targets : DiscoAgentOutput
            Generated targets `(π̂, ŷ, ẑ)`
        disco_h_state : chex.Array
            Updated Disco network hidden state
        meta_h_state : chex.Array
            Updated Meta-LNN hidden state
        """
        # Condition based on functional approach
        if params is not None:
            encoder, disco_net, meta_lnn, meta_proj = self.merge_params(params)
        else:
            encoder, disco_net, meta_lnn, meta_proj = (
                self.encoder,
                self.disco_net,
                self.meta_lnn,
                self.meta_proj,
            )

        # embed: (B, T, E), act_embed: (B, T, A, C)
        embedding, action_embed = encoder(rollout, action_mask=action_mask)
        disco_input = embedding

        # Apply meta conditioning from previous update (if available)
        if meta_h_state is not None:
            disco_input = self._meta_conditioning(meta_proj, embedding, meta_h_state)

        preds, disco_h_state = disco_net(
            disco_input,
            action_embed,
            h_state=disco_h_state,
        )

        # Update Meta-LNN for next iteration using trajectory summary
        summary = jnp.mean(embedding, axis=1)  # (B, T, E) -> (B, E)
        _, meta_h_state = meta_lnn(summary, h_state=meta_h_state)

        targets = DiscoAgentOutput(pi=preds.pi, y=preds.y, z=preds.z)

        # Freeze training
        if self.is_frozen:
            targets = jax.tree.map(jax.lax.stop_gradient, targets)
            disco_h_state = jax.lax.stop_gradient(disco_h_state)
            meta_h_state = jax.lax.stop_gradient(meta_h_state)

        return targets, disco_h_state, meta_h_state


class DiscoValueAgent(BaseAgent[ValueModules]):
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

    def __call__(
        self,
        obs: chex.Array,
        *,
        h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
        encoding: Optional[chex.Array] = None,
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

        encoding : jax.Array (optional)
            Pre-computed encoder features `(B, T, F)`. When provided,
            skips the encoder forward pass. Default is `None`

        Returns
        -------
        v : chex.Array
            State-value estimates with shape `(B, T, 1)` or `(B, 1)` if `T=1`
        h_state : chex.Array
            Updated hidden state. Shape: `(B, HS)`
        """
        # Reuse pre-computed encoding or compute fresh
        if encoding is None:
            encoding = self.encoder(obs)  # (B, T, F)

        # Compute state-value -> (B, T, 1)
        v, h_state = self.net(
            encoding,
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


class ContinuousDiscoAgent(BaseDiscoAgent[ContinuousDiscoModules]):
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

    def _build_modules(
        self,
        encoder_key: chex.PRNGKey,
        disco_key: chex.PRNGKey,
    ) -> None:
        self._encoder = ContinuousDiscoInputEncoder(
            self.encoder_config,
            self.config.max_action_dim,
            rngs=nnx.Rngs(encoder_key),
        )

        self._disco_net = ContinuousDiscoNetwork(
            self.encoder_config.output_dim,
            self.config.n_hidden,
            self.config.prediction_size,
            self.config.max_action_dim,
            key=disco_key,
            sparsity=self.config.sparsity,
        )

    def __call__(
        self,
        rollout: ContinuousRollout,
        *,
        disco_h_state: chex.Array | None = None,
        meta_h_state: chex.Array | None = None,
        params: nnx.State | None = None,
    ) -> Tuple[ContinuousDiscoAgentOutput, chex.Array, chex.Array]:
        """
        Generate targets for agent training.

        Parameters
        ----------
        rollout : ContinuousRollout
            A trajectory of experience
        disco_h_state : chex.Array (optional)
            Hidden state for DiscoNetwork. Shape: `(B, H)`.
            Default is `None`
        meta_h_state : chex.Array (optional)
            Hidden state for MetaLNN. Shape: `(B, H_meta)`.
            Default is `None`
        params : nnx.State (optional)
            Meta-agent parameters from `get_params()`. Only required if
            using a functional approach for `jax.grad`. Default is `None`

        Returns
        -------
        targets : ContinuousDiscoAgentOutput
            Generated targets `(μ̂, log σ̂, ŷ, ẑ)`
        disco_h_state : chex.Array
            Updated Disco network hidden state
        meta_h_state : chex.Array
            Updated Meta-LNN hidden state
        """
        # Condition based on functional approach
        if params is not None:
            encoder, disco_net, meta_lnn, meta_proj = self.merge_params(params)
        else:
            encoder, disco_net, meta_lnn, meta_proj = (
                self.encoder,
                self.disco_net,
                self.meta_lnn,
                self.meta_proj,
            )

        # embed: (B, T, E)
        embedding = encoder(rollout)
        disco_input = embedding

        # Apply meta conditioning from previous update (if available)
        if meta_h_state is not None:
            disco_input = self._meta_conditioning(meta_proj, embedding, meta_h_state)

        preds, disco_h_state = disco_net(disco_input, h_state=disco_h_state)

        # Update Meta-LNN for next iteration using trajectory summary
        summary = jnp.mean(embedding, axis=1)  # (B, T, E) -> (B, E)
        _, meta_h_state = meta_lnn(summary, h_state=meta_h_state)

        targets = ContinuousDiscoAgentOutput(
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


class ContinuousPolicyAgent(BasePolicyAgent[ContinuousPolicyModules]):
    """
    Creates a policy agent used to discover Reinforcement Learning (RL) rules.

    Combines DiscoRL target-based learning with Liquid Neural Networks (LNNs)
    and Gaussian policy parameterization. Suitable for continuous action spaces.

    Architecture:
        1. Input Encoder - extracts visual features from image observations
        2. Observation-Conditional Model (OCM) - processes encoded features
           to produce policy hidden representation and observation-conditioned
           predictions `(y)`
        3. Policy Head - projects OCM pi_hidden to Gaussian parameters
           `μ(s)` and `log σ(s)` at `max_action_dim` width
        4. Continuous Action-Conditional Model (ContinuousACM) - takes OCM
           embeddings concatenated with the continuous action taken, producing
           action-conditioned predictions `z(s, a)`, auxiliary policy
           predictions `aux_pi(s, a)`, and scalar Q-values `q(s, a)`
        5. Auxiliary Policy Head - projects ACM aux_pi hidden to predicted
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
    obs_spec : gym.spaces.Box
        A single observation space of the vectorized Gymnasium environment
    act_spec : gym.spaces.Box
        A single action space of the vectorized Gymnasium environment
    config : PolicyAgentSettings
        Configuration for the policy agent
    key : jax.random.PRNGKey
        Random number generator key
    max_action_dim : int
        Maximum continuous action dimensionality across all environments
        in the training set. Actions are zero-padded to this size
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Default is `False`
    """

    def __init__(
        self,
        obs_spec: gym.spaces.Box,
        act_spec: gym.spaces.Box,
        *,
        config: PolicyAgentSettings,
        key: chex.PRNGKey,
        max_action_dim: int,
        jit_compile: bool = False,
    ) -> None:
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.config = config
        self.key = key

        self.action_dim = int(np.prod(act_spec.shape))
        self.max_action_dim = max_action_dim

        # Action space bounds for clipping
        self.action_low = act_spec.low
        self.action_high = act_spec.high

        key_encoder, key_ocm, key_acm, key_decoder = jax.random.split(self.key, 4)

        # Observation encoder (image or vector, dispatched dynamically)
        self._encoder = build_obs_encoder(
            obs_spec.shape,
            config.n_hidden,
            key=key_encoder,
        )

        self._ocm = OCM(
            self._encoder.output_dim,
            config.n_hidden,
            config.prediction_size,
            key=key_ocm,
            sparsity=config.sparsity,
        )

        # ContinuousACM: takes (ocm_embedding, action) -> (z, aux_pi, q)
        self._acm = ContinuousACM(
            self._ocm.embedding_size,
            self.max_action_dim,
            config.n_hidden,
            config.prediction_size,
            key=key_acm,
            sparsity=config.sparsity,
        )

        # Policy decoder
        self._decoder = GaussianPolicyDecoder(
            config.prediction_size,
            max_action_dim,
            min_log_std=config.min_log_std,
            max_log_std=config.max_log_std,
            key=key_decoder,
        )

        self.encoder, self.ocm, self.acm, self.decoder = self._compile(jit_compile)

    def __call__(
        self,
        obs: chex.Array,
        action: chex.Array,
        *,
        ocm_h_state: Optional[chex.Array] = None,
        acm_h_state: Optional[chex.Array] = None,
        timespans: Optional[chex.Array] = None,
        encoding: Optional[chex.Array] = None,
    ) -> Tuple[ContinuousPolicyAgentOutput, PolicyAgentHiddenStates]:
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

        action : chex.Array
            Continuous action vector padded to `max_action_dim`.
            Used for ACM conditioning

        ocm_h_state : jax.Array (optional)
            Hidden state for the OCM `(B, HS)`. If `None`, initializes to zeros.
            Default is `None`

            - `batch_size (B)` the number of samples per timestep
            - `n_units (HS)` the total number of OCM hidden neurons

        acm_h_state : jax.Array (optional)
            Hidden state for the ACM `(B, HS)`. If `None`, initializes to zeros.
            Default is `None`

            - `batch_size (B)` the number of samples per timestep
            - `n_units (HS)` the total number of ACM hidden neurons

        timespans : jax.Array (optional)
            Time intervals between observations `(T,)`. If `None`, uses uniform
            time intervals. Default is `None`

            - `seq_length (T)` the number of sequences (e.g., trajectories)

        encoding : jax.Array (optional)
            Pre-computed encoder features `(B, T, F)`. When provided,
            skips the encoder forward pass. Default is `None`

        Returns
        -------
        preds : ContinuousPolicyAgentOutput
            An object of agent predictions
        h_state : PolicyAgentHiddenStates
            An object of the agents hidden states
        """
        # Reuse pre-computed encoding or compute fresh
        if encoding is None:
            encoding = self.encoder(obs)  # (B, T, F)

        # OCM forward - process observations and produce embeddings
        ocm_preds, ocm_h_state = self.ocm(
            encoding,
            h_state=ocm_h_state,
            timespans=timespans,
        )

        # ACM forward - use OCM embeddings for action-value predictions
        acm_preds, acm_h_state = self.acm(
            ocm_preds.embedding,
            action,
            h_state=acm_h_state,
            timespans=timespans,
        )

        # Policy decoding
        mu, log_std, aux_pi = self.decoder(
            ocm_preds.pi,
            acm_preds.aux_pi,
            action_dim_mask=jnp.arange(self.max_action_dim) < self.action_dim,
        )

        return (
            ContinuousPolicyAgentOutput.create(
                encoding,
                mu,
                log_std,
                ocm_preds.y,
                acm_preds.z,
                aux_pi,
                acm_preds.q,
            ),
            PolicyAgentHiddenStates(ocm=ocm_h_state, acm=acm_h_state),
        )

    def functional_forward(
        self,
        encoding: chex.Array,
        action: chex.Array,
        params: nnx.State,
        *,
        action_dim_mask: chex.Array | None = None,
    ) -> ContinuousPolicyAgentOutput:
        """
        Pure functional forward through OCM + ACM + Decoder
        with explicit params. Safe for use inside `jax.grad`.

        Parameters
        ----------
        encoding : chex.Array
           Encoder embeddings `(B, T, F)`

            - `batch_size (B)` the number of samples per timestep
            - `seq_length (T)` the number of sequences (e.g., trajectories)
            - `features (F)` number of features in the embedding
        action : chex.Array
            Continuous action vector `(B, T, D)` padded to `max_action_dim`
        params : nnx.State
            All trainable parameters
        action_dim_mask : chex.Array (optional)
            Boolean mask `(max_action_dim,)` for valid action dimensions.
            Default is `None`

        Returns
        -------
        preds : ContinuousPolicyAgentOutput
            Agent predictions
        """
        _, ocm, acm, decoder = self.merge_params(params)

        ocm_preds, _ = ocm(encoding)
        acm_preds, _ = acm(ocm_preds.embedding, action)

        if action_dim_mask is None:
            action_dim_mask = jnp.arange(self.max_action_dim) < self.action_dim

        mu, log_std, aux_pi = decoder(
            ocm_preds.pi,
            acm_preds.aux_pi,
            action_dim_mask=action_dim_mask,
        )

        return ContinuousPolicyAgentOutput.create(
            encoding,
            mu,
            log_std,
            ocm_preds.y,
            acm_preds.z,
            aux_pi,
            acm_preds.q,
        )

    def act(self, mu: chex.Array, log_std: chex.Array) -> np.ndarray:
        """
        Sample continuous actions from the Gaussian policy.

        Parameters
        ----------
        mu : chex.Array
            Policy mean `(B, D)` or `(B, T, D)`
        log_std : chex.Array
            Policy log standard deviation `(B, D)` or `(B, T, D)`

        Returns
        -------
        actions : np.Array
            Sampled actions `(B, max_action_dim)` clipped to action bounds
        """
        mu = np.asarray(mu)
        log_std = np.asarray(log_std)

        # Handle both squeezed (B, A) and non-squeezed (B, T, A) inputs
        if mu.ndim == 3:
            mu = mu.squeeze(axis=1)
            log_std = log_std.squeeze(axis=1)

        # Compute actions
        std = np.exp(log_std)
        noise = np.random.randn(*mu.shape).astype(np.float32)
        actions = mu + std * noise

        actions[:, : self.action_dim] = np.clip(
            actions[:, : self.action_dim],
            self.action_low,
            self.action_high,
        )
        actions[:, self.action_dim :] = 0.0
        return actions.astype(np.float32)
