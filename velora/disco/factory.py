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

import math
from typing import List, Tuple

import chex
import envrax
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from velora.disco.agent import DiscoAgent, PolicyAgent
from velora.disco.buffer import MixedBuffer
from velora.disco.config.metadata import PoolMetadata
from velora.disco.config.settings import (
    DiscoAgentSettings,
    PolicyAgentSettings,
    RuleTrainerSettings,
)
from velora.disco.config.spec import (
    DiscoSpec,
    DiscoValueSpec,
    EncoderSpec,
    PolicySpec,
    Specs,
)
from velora.disco.config.state import (
    AgentState,
    PolicySlot,
    PoolState,
    TargetSlot,
    ValueSlot,
)
from velora.disco.ema import EMAState
from velora.disco.nn.encoder import build_obs_encoder
from velora.disco.pool import pack_obs
from velora.disco.rollouts import Rollout
from velora.disco.utils import sample_budget
from velora.lnn.ncp import LNN
from velora.nn.optim import scale_by_adan_no_denom
from velora.utils.transforms import stack_pytrees


def create_optimizer(
    max_grad_norm: float,
    lr: float | None = None,
) -> optax.GradientTransformation:
    """
    Create optimizer chain for agent and value network training.

    Uses Adan optimizer (without denominator) with gradient clipping
    and scaled learning rate.

    Parameters
    ----------
    max_grad_norm : float
        Maximum gradient norm for gradient clipping
    lr : float (optional)
        Learning rate. When provided, scales updates by value. Default is `None`

    Returns
    -------
    optim : optax.GradientTransformation
        Configured optimizer chain
    """
    transforms = [scale_by_adan_no_denom(), optax.clip(max_grad_norm)]

    if lr is not None:
        transforms.append(optax.scale(-lr))

    return optax.chain(*transforms)


def create_encoder_spec(
    obs_shape: Tuple[int, ...],
    n_hidden: int,
    *,
    key: chex.PRNGKey,
    max_obs_dim: int,
) -> Tuple[EncoderSpec, nnx.State]:
    """
    Create an observation encoder spec for the rule trainer.

    Parameters
    ----------
    obs_shape : Tuple[int, ...]
        Observation shape
    n_hidden : int
        Number of hidden nodes
    key : chex.PRNGKey
        Random number generator key
    max_obs_dim : int
        Padded observation width size

    Returns
    -------
    spec : EncoderSpec
        Encoder specification
    params : nnx.State
        Encoder parameters
    """
    enc = build_obs_encoder(obs_shape, n_hidden, key, max_obs_dim)
    graphdef, params, rest = nnx.split(enc, nnx.Param, ...)

    spec = EncoderSpec(
        graphdef=graphdef,
        rest=rest,
        encoding_dim=enc.encoding_dim,
        max_obs_dim=max_obs_dim,
    )
    return spec, params


def create_policy_spec(
    encoding_dim: int,
    max_action_dim: int,
    config: PolicyAgentSettings,
    *,
    key: chex.PRNGKey,
) -> Tuple[PolicySpec, nnx.State]:
    """
    Create a policy agent spec for the rule trainer.

    Parameters
    ----------
    encoding_dim : int
        OCM encoding input dimension
    max_action_dim : int
        Maximum action dimensionality size
    config : PolicyAgentSettings
        Policy agent configuration settings
    key : chex.PRNGKey
        Random number generator key

    Returns
    -------
    spec : PolicySpec
        Policy agent specification
    params : nnx.State
        Agent parameters
    """
    p = PolicyAgent(encoding_dim, max_action_dim, config=config, key=key)
    graphdef, params, rest = nnx.split(p, nnx.Param, ...)

    spec = PolicySpec(
        graphdef=graphdef,
        rest=rest,
        ocm_hidden_size=p.ocm.hidden_size,
        acm_hidden_size=p.acm.hidden_size,
        max_action_dim=max_action_dim,
    )
    return spec, params


def create_disco_spec(
    max_action_dim: int,
    config: DiscoAgentSettings,
    *,
    key: chex.PRNGKey,
) -> Tuple[DiscoSpec, nnx.State]:
    """
    Create a disco agent spec for the rule trainer.

    Parameters
    ----------
    max_action_dim : int
        Maximum action dimensionality size
    config : DiscoAgentSettings
        Disco agent configuration settings
    key : chex.PRNGKey
        Random number generator key

    Returns
    -------
    spec : DiscoSpec
        Disco agent specification
    params : nnx.State
        Agent parameters
    """
    d = DiscoAgent(max_action_dim, config=config, key=key)
    graphdef, params, rest = nnx.split(d, nnx.Param, ...)

    spec = DiscoSpec(
        graphdef=graphdef,
        rest=rest,
        disco_hidden_size=d.disco_net.hidden_size,
        meta_hidden_size=d.meta_lnn.hidden_size,
        is_frozen=d.is_frozen,
    )
    return spec, params


def create_disco_value_spec(
    encoding_dim: int,
    n_hidden: int,
    *,
    key: chex.PRNGKey,
) -> Tuple[DiscoValueSpec, nnx.State]:
    """
    Create a disco value agent spec for the rule trainer.

    Parameters
    ----------
    encoding_dim : int
        Number of input nodes
    n_hidden : int
        Number of hidden nodes
    key : chex.PRNGKey
        Random number generator key

    Returns
    -------
    spec : DiscoValueSpec
        Disco value agent specification
    params : nnx.State
        Agent parameters
    """
    v = LNN(encoding_dim, n_hidden, out_features=1, key=key)
    graphdef, params, rest = nnx.split(v, nnx.Param, ...)

    spec = DiscoValueSpec(
        graphdef=graphdef,
        rest=rest,
        hidden_size=v.hidden_size,
    )
    return spec, params


def create_specs(
    obs_shape: Tuple[int, ...],
    max_action_dim: int,
    max_obs_dim: int,
    config: RuleTrainerSettings,
    *,
    key: chex.PRNGKey,
) -> Tuple[Specs, nnx.State]:
    """
    Create rule trainer agent static metadata as specs and the
    disco agent parameters.

    Parameters
    ----------
    obs_shape : Tuple[int, ...]
        Observation shape used to size the encoder
    max_action_dim : int
        Padded action width shared across all trainer slots
    max_obs_dim : int
        Padded observation width shared across all trainer slots
    config : RuleTrainerSettings
        Rule trainer configuration settings
    key : chex.PRNGKey
        Random number generator key

    Returns
    -------
    specs : Specs
        Bundle of shared rule trainer agent specs
    disco_params : nnx.State
        Initial disco agent parameters
    """
    k_enc, k_pol, k_dis, k_val = jax.random.split(key, 4)

    enc_spec, _ = create_encoder_spec(
        obs_shape,
        config.agent.n_hidden,
        key=k_enc,
        max_obs_dim=max_obs_dim,
    )
    enc_dim = enc_spec.encoding_dim

    pol_spec, _ = create_policy_spec(
        enc_dim,
        max_action_dim,
        config.agent,
        key=k_pol,
    )
    disco_spec, disco_params = create_disco_spec(
        max_action_dim,
        config.disco_agent,
        key=k_dis,
    )
    value_spec, _ = create_disco_value_spec(
        enc_dim,
        config.agent.n_hidden,
        key=k_val,
    )

    specs = Specs(
        encoder=enc_spec,
        policy=pol_spec,
        disco=disco_spec,
        value=value_spec,
    )
    return specs, disco_params


def create_trainer_slot(
    obs_space: envrax.Box,
    action_space: envrax.Box,
    *,
    config: RuleTrainerSettings,
    key: chex.PRNGKey,
    specs: Specs,
    disco_params: nnx.State,
    max_action_dim: int,
    max_obs_dim: int,
) -> AgentState:
    """
    Creates an agent trainer slot for a single environment.

    Parameters
    ----------
    obs_space : envrax.Box
        Environment observation space
    action_space : envrax.Box
        Environment action space
    config : RuleTrainerSettings
        Rule trainer configuration settings
    key : chex.PRNGKey
        Random number generator key
    specs : Specs
        Rule trainer specifications
    disco_params : nnx.State
        Disco agent parameters
    max_action_dim : int
        Maximum action dimensionality size
    max_obs_dim : int
        Padded observation width size

    Returns
    -------
    state : AgentState
        Agent trainer state
    """
    k_enc, k_pol, k_tgt, k_val, k_budget = jax.random.split(key, 5)
    enc_dim = specs.encoder.encoding_dim

    _, enc_params = create_encoder_spec(
        obs_space.shape,
        config.agent.n_hidden,
        key=k_enc,
        max_obs_dim=max_obs_dim,
    )
    _, pol_params = create_policy_spec(
        enc_dim,
        max_action_dim,
        config.agent,
        key=k_pol,
    )
    _, tgt_params = create_policy_spec(
        enc_dim,
        max_action_dim,
        config.agent,
        key=k_tgt,
    )
    _, val_params = create_disco_value_spec(
        enc_dim,
        config.agent.n_hidden,
        key=k_val,
    )

    enc_opt = create_optimizer(config.agent.max_grad_norm, lr=config.agent.lr)
    pol_opt = create_optimizer(config.agent.max_grad_norm, lr=config.agent.lr)
    val_opt = create_optimizer(
        config.disco_value.max_grad_norm,
        lr=config.disco_value.lr,
    )
    meta_opt = create_optimizer(config.meta_grad_norm)

    action_dim = int(math.prod(action_space.shape))

    return AgentState(
        encoder_params=enc_params,
        encoder_opt=enc_opt.init(enc_params),
        policy=PolicySlot(
            params=pol_params,
            opt_state=pol_opt.init(pol_params),
            ocm_h=jnp.zeros((1, specs.policy.ocm_hidden_size)),
            acm_h=jnp.zeros((1, specs.policy.acm_hidden_size)),
        ),
        target=TargetSlot(
            params=tgt_params,
            ocm_h=jnp.zeros((1, specs.policy.ocm_hidden_size)),
            acm_h=jnp.zeros((1, specs.policy.acm_hidden_size)),
        ),
        value=ValueSlot(
            params=val_params,
            opt_state=val_opt.init(val_params),
            hidden=jnp.zeros((1, specs.value.hidden_size)),
        ),
        meta_opt_state=meta_opt.init(disco_params),
        adv_ema=EMAState.create(),
        td_ema=EMAState.create(),
        action_mask=jnp.arange(max_action_dim) < action_dim,
        env_steps=jnp.int32(0),
        step_budget=jnp.int32(sample_budget(k_budget)),
        collection_steps=jnp.int32(0),
        obs=jnp.zeros((1, max_obs_dim)),
        prev_action=jnp.zeros((1, max_action_dim)),
    )


def create_pool_state(
    slot_inits: List[AgentState],
    multi_env: envrax.MultiEnv,
    meta: PoolMetadata,
    config: RuleTrainerSettings,
    encoder_spec: EncoderSpec,
    *,
    key: chex.PRNGKey,
) -> PoolState:
    """
    Create a trainer pool state and initialize the multi-environments.

    Parameters
    ----------
    slot_inits : List[AgentState]
        Per-slot agent trainer states
    multi_env : envrax.MultiEnv
        Multi-environment instance for all agent trainers
    meta : PoolMetadata
        Pool trainer metadata
    config : RuleTrainerSettings
        Rule trainer configuration settings
    encoder_spec : EncoderSpec
        Encoder specification
    key : chex.PRNGKey
        Random number generator key

    Returns
    -------
    state : PoolState
        Pool trainer state
    """
    env_key, pool_key = jax.random.split(key, 2)

    # Stack agent states - (P, ...)
    slots: AgentState = stack_pytrees(slot_inits)  # type: ignore

    obs_dict, env_states = multi_env.reset(env_key)
    initial_obs = pack_obs(obs_dict, meta)  # (P, 1, max_obs_dim)

    slots = slots.__replace__(obs=initial_obs)

    buf_kwargs = dict(
        n_actions=meta.max_action_dim,
        encoding_dim=encoder_spec.encoding_dim,
        prediction_dim=config.agent.prediction_size,
    )
    train_buffer = MixedBuffer.create(
        num_trainers=meta.num_trainers,
        capacity=config.replay_capacity,
        replay_ratio=config.replay_ratio,
        seq_len=config.seq_len,
        **buf_kwargs,  # type: ignore
    )
    valid_rollout = Rollout.zeros(
        shape_prefix=(meta.num_trainers, 1, config.seq_len * 2),
        **buf_kwargs,  # type: ignore
    )

    return PoolState(
        slots=slots,
        env_states=env_states,
        train_buffer=train_buffer,
        valid_rollout=valid_rollout,
        rng=pool_key,
    )
