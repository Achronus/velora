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

from typing import Self, Tuple

import chex
import jax.numpy as jnp
import optax
from flax import struct

from velora.disco.ema import EMAState

HiddenState = chex.Array | None


@struct.dataclass
class PolicyAgentHiddenStates:
    """
    Dataclass for `PolicyAgent` hidden states.

    Parameters
    ----------
    ocm : jax.Array
        Observation-Conditional Model (OCM) hidden states `(B, H)`

        - `batch_size (B)` the number of samples per timestep
        - `n_units (H)` the total number of OCM hidden neurons

    acm : jax.Array
        Action-Conditional Model (ACM) hidden states `(B * A, H)`

        - `batch_size (B)` the number of samples per timestep
        - `n_actions (A)` the number of discrete actions
        - `n_units (H)` the total number of ACM hidden neurons
    """

    ocm: chex.Array
    acm: chex.Array


@struct.dataclass
class AgentTrainerHiddenStates:
    """
    Dataclass for `AgentTrainer` hidden states.

    Parameters
    ----------
    policy_ocm : chex.Array (optional)
        Policy network OCM hidden state. Shape: `(B, H)`. Default is `None`
    policy_acm : chex.Array (optional)
        Policy network ACM hidden state. Shape: `(B*A, H)`. Default is `None`
    target_ocm : chex.Array (optional)
        Target network OCM hidden state. Shape: `(B, H)`. Default is `None`
    target_acm : chex.Array (optional)
        Target network ACM hidden state. Shape: `(B*A, H)`. Default is `None`
    value : chex.Array (optional)
        Value network hidden state. Shape: `(B, H)`. Default is `None`
    """

    policy_ocm: HiddenState = None
    policy_acm: HiddenState = None
    target_ocm: HiddenState = None
    target_acm: HiddenState = None
    value: HiddenState = None

    def reset_on_done(self, discounts: chex.Array, n_actions: int) -> Self:
        """
        Reset hidden states where episodes terminated (`discount=0`).

        Applies a continuation mask to all hidden states. Where:
            - `discount=1.0` - the hidden state is unchanged
            - `discount=0.0` - the hidden state is zeroed, resetting the
            recurrent context for that environment

        Parameters
        ----------
        discounts : chex.Array
            Episode dones from the environment `(B, 1)`
        n_actions : int
            Number of actions agent can take

        Returns
        -------
        new_state : AgentTrainerHiddenStates
            Updated state
        """
        # Squeeze to (B,) for broadcasting against hidden states (B, H)
        mask = jnp.squeeze(discounts)  # (B,)
        mask_expanded = jnp.repeat(mask, n_actions)  # (B*A,)

        def _apply_mask(h: chex.Array | None, m: chex.Array) -> chex.Array | None:
            return h * m[:, None] if h is not None else None  # type: ignore

        mask_expanded = jnp.repeat(mask, n_actions)

        return self.__class__(
            policy_ocm=_apply_mask(self.policy_ocm, mask),
            policy_acm=_apply_mask(self.policy_acm, mask_expanded),
            target_ocm=_apply_mask(self.target_ocm, mask),
            target_acm=_apply_mask(self.target_acm, mask_expanded),
            value=_apply_mask(self.value, mask),
        )

    def update(
        self,
        h_state: PolicyAgentHiddenStates,
        target_h_state: PolicyAgentHiddenStates,
        value_h_state: chex.Array,
    ) -> Self:
        """
        Update hidden states from forward pass outputs.

        Parameters
        ----------
        h_state : AgentHiddenStates
            Policy network hidden states
        target_h_state : AgentHiddenStates
            Target network hidden states
        value_h_state : chex.Array
            Value network hidden state

        Returns
        -------
        new_state : AgentTrainerHiddenStates
            Updated state
        """
        return self.__class__(
            policy_ocm=h_state.ocm,
            policy_acm=h_state.acm,
            target_ocm=target_h_state.ocm,
            target_acm=target_h_state.acm,
            value=value_h_state,
        )


@struct.dataclass
class RuleTrainerHiddenStates:
    """
    Dataclass for `RuleTrainer` hidden states.

    Maintains hidden states for the `DiscoNetwork` and `MetaLNN` across
    all environments during meta-training.

    Make a new instance using the `.create()` method.

    Parameters
    ----------
    disco : Tuple[chex.Array, ...]
        `DiscoNetwork` hidden states for each environment
    meta : Tuple[chex.Array, ...]
        `MetaLNN` hidden states for each environment
    disco_zero : chex.Array
        Empty template array for resetting
    meta_zero : chex.Array
        Empty template array for resetting
    """

    disco: Tuple[chex.Array, ...]
    meta: Tuple[chex.Array, ...]

    disco_zero: chex.Array
    meta_zero: chex.Array

    @classmethod
    def create(
        cls,
        num_envs: int,
        batch_size: int,
        disco_h_size: int,
        meta_h_size: int,
    ) -> Self:
        """
        Create initial hidden states for all environments.

        Parameters
        ----------
        num_envs : int
            Number of environments in the population
        batch_size : int
            Batch size (num_vec_envs)
        disco_h_size : int
            Disco network hidden size
        meta_h_size : int
            Meta LNN hidden size

        Returns
        -------
        states : RuleTrainerHiddenStates
            Initialized hidden states (all `0s`)
        """
        disco_zero = jnp.zeros((batch_size, disco_h_size))
        meta_zero = jnp.zeros((batch_size, meta_h_size))

        return cls(
            disco=tuple(disco_zero for _ in range(num_envs)),
            meta=tuple(meta_zero for _ in range(num_envs)),
            disco_zero=disco_zero,
            meta_zero=meta_zero,
        )

    def update(self, env_idx: int, disco_h: HiddenState, meta_h: HiddenState) -> Self:
        """
        Update hidden states for a specific environment.

        Parameters
        ----------
        env_idx : int
            Index of the environment to update
        disco_h : chex.Array | None
            New disco network hidden state. When `None` replaced with array of `0s`
        meta_h : chex.Array | None
            New meta-LNN hidden state. When `None` replaced with array of `0s`

        Returns
        -------
        new_state : RuleTrainerHiddenStates
            An updated hidden state
        """
        disco_list = list(self.disco)
        meta_list = list(self.meta)

        disco_list[env_idx] = disco_h if disco_h is not None else self.disco_zero
        meta_list[env_idx] = meta_h if meta_h is not None else self.meta_zero

        return self.__replace__(
            disco=tuple(disco_list),
            meta=tuple(meta_list),
        )

    def reset(self, env_idx: int) -> Self:
        """
        Reset hidden states for a specific environment.

        Parameters
        ----------
        env_idx : int
            Index of the environment to reset

        Returns
        -------
        new_state : RuleTrainerHiddenStates
            An updated hidden state with single environments reset to `0s`
        """
        return self.update(env_idx, None, None)

    def get(self, env_idx: int) -> Tuple[chex.Array, chex.Array]:
        """
        Get hidden states for a specific environment.

        Parameters
        ----------
        env_idx : int
            Index of the environment

        Returns
        -------
        disco_h : chex.Array
            An environments Disco network hidden state
        meta_h : chex.Array
            An environments Meta-LNN hidden state
        """
        return self.disco[env_idx], self.meta[env_idx]


@struct.dataclass
class AgentTrainerState:
    """
    Dataclass for `AgentTrainer` state.

    Make a new instance using the `AgentTrainerState.create()` method.

    Parameters
    ----------
    policy_opt_state : optax.OptState
        Policy optimizer state
    value_opt_state : optax.OptState
        Value optimizer state
    hidden : AgentHiddenStates
        Recurrent hidden states
    adv_ema : EMAState
        EMA state for advantage normalization
    td_ema : EMAState
        EMA state for TD error normalization
    current_obs : chex.Array
        Current observation from environment
    steps_trained : int (optional)
        Total training steps completed. Default is `0`
    """

    policy_opt_state: optax.OptState
    value_opt_state: optax.OptState

    hidden: AgentTrainerHiddenStates
    adv_ema: EMAState
    td_ema: EMAState

    current_obs: chex.Array
    steps_trained: int = 0

    @classmethod
    def create(
        cls,
        policy_opt_state: optax.OptState,
        value_opt_state: optax.OptState,
        current_obs: chex.Array,
    ) -> Self:
        """
        Creates a new state instance.

        Parameters
        ----------
        policy_opt_state : optax.OptState
            Initialized policy optimizer state
        value_opt_state : optax.OptState
            Initialized value optimizer state
        current_obs : chex.Array
            Initial observation from environment

        Returns
        -------
        new_state : AgentTrainerState
            Initialized state
        """
        return cls(
            policy_opt_state=policy_opt_state,
            value_opt_state=value_opt_state,
            hidden=AgentTrainerHiddenStates(),
            adv_ema=EMAState.create(),
            td_ema=EMAState.create(),
            current_obs=current_obs,
            steps_trained=0,
        )

    def update_policy_opt(self, new_opt_state: optax.OptState) -> Self:
        """
        Update policy optimizer state.

        Parameters
        ----------
        new_optim_state : optax.OptState
            New agent optimizer state

        Returns
        -------
        state : AgentTrainerState
            A new state with updated `policy_opt_state` and `steps_trained + 1`
        """
        return self.__replace__(
            policy_opt_state=new_opt_state,
            steps_trained=self.steps_trained + 1,
        )

    def update_value_opt(self, new_opt_state: optax.OptState) -> Self:
        """
        Update value optimizer state.

        Parameters
        ----------
        new_optim_state : optax.OptState
            New value optimizer state

        Returns
        -------
        state : AgentTrainerState
            A new state with updated `value_opt_state`
        """
        return self.__replace__(value_opt_state=new_opt_state)

    def update_hidden(self, new_hidden: AgentTrainerHiddenStates) -> Self:
        """
        Update recurrent hidden states.

        Parameters
        ----------
        new_hidden : BundledHiddenStates
            New hidden states

        Returns
        -------
        state : AgentTrainerState
            A new state with updated `hidden`
        """
        return self.__replace__(hidden=new_hidden)

    def update_ema(self, adv_ema: EMAState, td_ema: EMAState) -> Self:
        """
        Update value state.

        Parameters
        ----------
        new_state : DiscoValueState
            New value state

        Returns
        -------
        state : AgentTrainerState
            A new state with updated `adv_ema`, `td_ema`
        """
        return self.__replace__(adv_ema=adv_ema, td_ema=td_ema)

    def update_obs(self, new_obs: chex.Array) -> Self:
        """
        Update observation state.

        Parameters
        ----------
        new_obs : chex.Array
            New current observation

        Returns
        -------
        state : AgentTrainerState
            New state with updated `current_obs`
        """
        return self.__replace__(current_obs=new_obs)


@struct.dataclass
class RuleTrainerState:
    """
    Dataclass for `RuleTrainer` checkpoint state.

    Make a new instance using the `RuleTrainerState.create()` method.

    Parameters
    ----------
    meta_opt_state : optax.OptState
        Disco-optimizer state
    hidden : RuleTrainerHiddenStates
        Rule Trainer hidden states per environment
    meta_step : int (optional)
        Current meta-training step. Default is `0`
    """

    meta_opt_state: optax.OptState
    hidden: RuleTrainerHiddenStates
    meta_step: int = 0

    @classmethod
    def create(
        cls,
        meta_opt_state: optax.OptState,
        num_envs: int,
        batch_size: int,
        disco_h_size: int,
        meta_h_size: int,
    ) -> Self:
        """
        Create initial meta-training state.

        Parameters
        ----------
        meta_opt_state : optax.OptState
            Disco network optimizer state
        num_envs : int
            Number of environments
        batch_size : int
            Batch size (num_vec_envs)
        disco_h_size : int
            Disco network hidden size
        meta_h_size : int
            Meta LNN hidden size

        Returns
        -------
        state : RuleTrainerState
            Initialized state
        """
        return cls(
            meta_opt_state=meta_opt_state,
            hidden=RuleTrainerHiddenStates.create(
                num_envs,
                batch_size,
                disco_h_size,
                meta_h_size,
            ),
            meta_step=0,
        )

    def update_opt(self, new_opt_state: optax.OptState) -> Self:
        """
        Update meta-optimizer state.

        Parameters
        ----------
        new_opt_state : optax.OptState
            New disco network optimizer state

        Returns
        -------
        state : RuleTrainerState
            New state with updated `meta_opt_state` and `meta_step + 1`
        """
        return self.__replace__(
            meta_opt_state=new_opt_state,
            meta_step=self.meta_step + 1,
        )

    def update_hidden(
        self,
        env_idx: int,
        disco_h: HiddenState,
        meta_h: HiddenState,
    ) -> Self:
        """
        Update hidden states for specific environment.

        Parameters
        ----------
        env_idx : int
            Index of the environment to update
        disco_h : HiddenState
            Updated Disco Network hidden state
        meta_h : HiddenState
            Updated Meta LNN hidden state

        Returns
        -------
        state : RuleTrainerState
            New state with updated `hidden`
        """
        return self.__replace__(
            hidden=self.hidden.update(env_idx, disco_h, meta_h),
        )
