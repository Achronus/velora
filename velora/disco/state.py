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

from typing import Any, Dict, Self, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax import struct

from velora.base.rollouts import Rollout
from velora.disco.ema import EMAState
from velora.disco.outputs import PolicyAgentOutput

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

        Parameters
        ----------
        discounts : chex.Array
            Episode dones from the environment
        n_actions : int
            Number of actions agent can take

        Returns
        -------
        new_state : AgentTrainerHiddenStates
            Updated state
        """
        if not jnp.any(discounts == 0.0):
            return self

        # Ensure discounts is 1D: (B,)
        mask = jnp.squeeze(discounts)

        def _end_check(h: chex.Array | None, m: chex.Array) -> chex.Array | None:
            return h * m[:, None] if h is not None else None  # type: ignore

        mask_expanded = jnp.repeat(mask, n_actions)

        return self.__class__(
            policy_ocm=_end_check(self.policy_ocm, mask),
            policy_acm=_end_check(self.policy_acm, mask_expanded),
            target_ocm=_end_check(self.target_ocm, mask),
            target_acm=_end_check(self.target_acm, mask_expanded),
            value=_end_check(self.value, mask),
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
    disco : Tuple[chex.Array | None, ...]
        `DiscoNetwork` hidden states for each environment
    meta : Tuple[chex.Array | None, ...]
        `MetaLNN` hidden states for each environment
    """

    disco: Tuple[HiddenState, ...]
    meta: Tuple[HiddenState, ...]

    @classmethod
    def create(cls, num_envs: int) -> Self:
        """
        Create initial hidden states for all environments.

        Parameters
        ----------
        num_envs : int
            Number of environments in the population

        Returns
        -------
        states : RuleTrainerHiddenStates
            Initialized hidden states (all `None`)
        """
        return cls(
            disco=tuple(None for _ in range(num_envs)),
            meta=tuple(None for _ in range(num_envs)),
        )

    def update(self, env_idx: int, disco_h: HiddenState, meta_h: HiddenState) -> Self:
        """
        Update hidden states for a specific environment.

        Parameters
        ----------
        env_idx : int
            Index of the environment to update
        disco_h : chex.Array | None
            New disco network hidden state
        meta_h : chex.Array | None
            New meta-LNN hidden state

        Returns
        -------
        new_state : RuleTrainerHiddenStates
            An updated hidden state
        """
        disco_list = list(self.disco)
        meta_list = list(self.meta)

        disco_list[env_idx] = disco_h
        meta_list[env_idx] = meta_h

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
            An updated hidden state with a single environments reset to `None`
        """
        return self.update(env_idx, None, None)

    def get(self, env_idx: int) -> Tuple[HiddenState, HiddenState]:
        """
        Get hidden states for a specific environment.

        Parameters
        ----------
        env_idx : int
            Index of the environment

        Returns
        -------
        disco_h : chex.Array | None
            An environments Disco network hidden state
        meta_h : chex.Array | None
            An environments Meta-LNN hidden state
        """
        return self.disco[env_idx], self.meta[env_idx]


@struct.dataclass
class CollectState:
    """
    Dataclass for the `AgentTrainer.collect()` method.

    Parameters
    ----------
    actions : Tuple[chex.Array, ...] (optional)
        Actions for a trajectory. Default is `()`
    rewards : Tuple[chex.Array, ...] (optional)
        Rewards for a trajectory. Default is `()`
    discounts : Tuple[chex.Array, ...] (optional)
        Discounts for a trajectory. Default is `()`
    value : Tuple[chex.Array, ...] (optional)
        State-value estimates for a trajectory. Default is `()`
    preds : Tuple[AgentOutput, ...] (optional)
        Policy network predictions for a trajectory.
        Default is `()`
    target_preds : Tuple[AgentOutput, ...] (optional)
        Target network predictions for a trajectory.
        Default is `()`
    completed_returns : Tuple[float, ...] (optional)
        Episodic returns for a trajectory. Default is `()`
    completed_lengths : Tuple[int, ...] (optional)
        Episode lengths for a a trajectory. Default is `()`
    """

    actions: Tuple[chex.Array, ...] = ()
    rewards: Tuple[chex.Array, ...] = ()
    discounts: Tuple[chex.Array, ...] = ()
    values: Tuple[chex.Array, ...] = ()
    preds: Tuple[PolicyAgentOutput, ...] = ()
    target_preds: Tuple[PolicyAgentOutput, ...] = ()
    completed_returns: Tuple[float, ...] = ()
    completed_lengths: Tuple[int, ...] = ()

    def append_step(
        self,
        actions: chex.Array,
        rewards: chex.Array,
        discounts: chex.Array,
        values: chex.Array,
        preds: PolicyAgentOutput,
        target_preds: PolicyAgentOutput,
    ) -> Self:
        """
        Append a single timestep of data.

        Parameters
        ----------
        actions : chex.Array
            Actions taken in the environment
        rewards : chex.Array
            Rewards obtained from the environment
        discounts : chex.Array
            Done status from the environment
        values : chex.Array
            State-value estimate predictions at timestep
        preds : AgentOutput
            Policy network predictions at timestep
        target_preds : AgentOutput
            Target network predictions at timestep

        Returns
        -------
        state : Self
            An updated state
        """
        return self.__replace__(
            actions=self.actions + (actions,),
            rewards=self.rewards + (rewards,),
            discounts=self.discounts + (discounts,),
            values=self.values + (values,),
            preds=self.preds + (preds,),
            target_preds=self.target_preds + (target_preds,),
        )

    def record_episodes(self, info: Dict[str, Any]) -> Self:
        """
        Record completed episode statistics from env info.

        Parameters
        ----------
        info : Dict[str, Any]
            Environment metadata

        Returns
        -------
        state : Self
            An updated state
        """
        if "final_info" not in info:
            return self

        new_returns = self.completed_returns
        new_lengths = self.completed_lengths

        for env_info in info["final_info"]:
            if env_info is not None and "episode" in env_info:
                new_returns = new_returns + (env_info["episode"]["r"],)
                new_lengths = new_lengths + (env_info["episode"]["l"],)

        return self.__replace__(
            completed_returns=new_returns,
            completed_lengths=new_lengths,
        )

    def to_rollout(self) -> Rollout:
        """
        Stack sequence data into a rollout.

        Returns
        -------
        rollout : Rollout
            Trajectory of samples
        """
        actions_batch = jnp.stack(self.actions, axis=1)
        rewards_batch = jnp.stack(self.rewards, axis=1)
        discounts_batch = jnp.stack(self.discounts, axis=1)
        values_batch = jnp.stack(self.values, axis=1)

        batch_preds = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=1),
            *self.preds,
        )
        target_batch_preds = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=1),
            *self.target_preds,
        )

        return Rollout(
            actions=actions_batch,
            rewards=rewards_batch,
            discounts=discounts_batch,
            values=values_batch,
            preds=batch_preds,
            target_preds=target_batch_preds,
        )

    def episode_metrics(self) -> Dict[str, float] | None:
        """
        Return episode metrics if any episodes completed.

        Returns
        -------
        metrics : Dict[str, float] | None
            Episode metrics dict, or `None` if no episodes completed
        """
        if not self.completed_returns:
            return None

        return {
            "episode/reward_mean": self.mean_episode_return,
            "episode/reward_min": float(min(self.completed_returns)),
            "episode/reward_max": float(max(self.completed_returns)),
            "episode/length_mean": self.mean_episode_length,
            "episode/count": self.num_completed_episodes,
        }

    @property
    def num_completed_episodes(self) -> int:
        return len(self.completed_returns)

    @property
    def total_episode_reward(self) -> float:
        if not self.completed_returns:
            return 0.0

        return sum(self.completed_returns)

    @property
    def mean_episode_return(self) -> float:
        if not self.completed_returns:
            return 0.0

        return sum(self.completed_returns) / len(self.completed_returns)

    @property
    def mean_episode_length(self) -> float:
        if not self.completed_lengths:
            return 0.0

        return sum(self.completed_lengths) / len(self.completed_lengths)


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
    ) -> Self:
        """
        Create initial meta-training state.

        Parameters
        ----------
        meta_opt_state : optax.OptState
            Disco network optimizer state
        num_envs : int
            Number of environments

        Returns
        -------
        state : RuleTrainerState
            Initialized state
        """
        return cls(
            meta_opt_state=meta_opt_state,
            hidden=RuleTrainerHiddenStates.create(num_envs),
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
