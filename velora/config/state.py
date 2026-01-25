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

from typing import TYPE_CHECKING, Any, Dict, Self, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct

from velora.config.outputs import BufferSamples, PolicyAgentOutput

if TYPE_CHECKING:
    from velora.metrics.ema import MovingAverage


@struct.dataclass
class EMAState:
    """
    Exponential Moving Average (EMA) state.

    Parameters
    ----------
    moment1 : jax.ArrayTree
        The first set of moments
    moment2 : jax.ArrayTree
        The second set of moments
    decay_product : jax.Array
        The product of all decays from start of accumulation
    """

    moment1: chex.ArrayTree
    moment2: chex.ArrayTree
    decay_product: chex.Array


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
class BundledHiddenStates:
    """
    Dataclass for bundled hidden states for the `AgentTrainer` used during data collection.

    Parameters
    ----------
    ocm : chex.Array (optional)
        Policy network OCM hidden state. Default is `None`
    acm : chex.Array (optional)
        Policy network ACM hidden state. Default is `None`
    target_ocm : chex.Array (optional)
        Target network OCM hidden state. Default is `None`
    target_acm : chex.Array (optional)
        Target network ACM hidden state. Default is `None`
    """

    ocm: chex.Array | None = None
    acm: chex.Array | None = None
    target_ocm: chex.Array | None = None
    target_acm: chex.Array | None = None

    def reset_on_done(self, discounts: chex.Array, n_actions: int) -> None:
        """
        Reset hidden states where episodes terminated (`discount=0`).

        Parameters
        ----------
        discounts : chex.Array
            Episode dones from the environment
        n_actions : int
            Number of actions agent can take
        """
        if not jnp.any(discounts == 0.0):
            return

        def _end_check(h: chex.Array | None, mask: chex.Array) -> chex.Array | None:
            return h * mask[:, None] if h is not None else None  # type: ignore

        discounts_expanded = jnp.repeat(discounts, n_actions)
        self = self.__replace__(
            ocm=_end_check(self.ocm, discounts),
            acm=_end_check(self.acm, discounts_expanded),
            target_ocm=_end_check(self.target_ocm, discounts),
            target_acm=_end_check(self.target_acm, discounts_expanded),
        )

    def update(
        self,
        h_state: PolicyAgentHiddenStates,
        target_h_state: PolicyAgentHiddenStates,
    ) -> None:
        """
        Update hidden states from forward pass outputs.

        Parameters
        ----------
        h_state : AgentHiddenStates
            Policy network hidden states
        target_h_state : AgentHiddenStates
            Target network hidden states
        """
        self = self.__replace__(
            ocm=h_state.ocm,
            acm=h_state.acm,
            target_ocm=target_h_state.ocm,
            target_acm=target_h_state.acm,
        )


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
    preds: Tuple[PolicyAgentOutput, ...] = ()
    target_preds: Tuple[PolicyAgentOutput, ...] = ()
    completed_returns: Tuple[float, ...] = ()
    completed_lengths: Tuple[int, ...] = ()

    def append_step(
        self,
        actions: chex.Array,
        rewards: chex.Array,
        discounts: chex.Array,
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

    def to_batches(self) -> BufferSamples:
        """
        Stack collected data into batched arrays `(B, T)`.

        Returns
        -------
        samples : BufferSamples
            Trajectory of samples
        """
        actions_batch = jnp.stack(self.actions, axis=1)
        rewards_batch = jnp.stack(self.rewards, axis=1)
        discounts_batch = jnp.stack(self.discounts, axis=1)

        batch_preds = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *self.preds)
        target_batch_preds = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=1), *self.target_preds
        )

        return BufferSamples(
            actions=actions_batch,
            rewards=rewards_batch,
            discounts=discounts_batch,
            preds=batch_preds,
            target_preds=target_batch_preds,
        )

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
class ParameterState:
    """
    Dataclass for `AgentTrainer` parameter states.

    Parameters
    ----------
    policy : nnx.State
        Policy agent network parameters
    target : nnx.State
        Target network parameters (EMA of policy)
    """

    policy: nnx.State
    target: nnx.State


@struct.dataclass
class DiscoValueState:
    """
    Dataclass for `DiscoValueAgent` state.

    Set a new instance with `DiscoValueAgent.create()` method.

    Parameters
    ----------
    params : nnx.State
        Value network parameters
    optim : optax.GradientTransformation
        Value network optimizer
    opt_state : optax.OptState
        Value network optimizer state
    adv_ema : MovingAverage
        EMA for advantage normalization
    td_ema : MovingAverage
        EMA for TD error normalization
    """

    params: nnx.State
    optim: optax.GradientTransformation
    opt_state: optax.OptState

    adv_ema: "MovingAverage"
    td_ema: "MovingAverage"

    @classmethod
    def create(
        cls,
        params: nnx.State,
        optim: optax.GradientTransformation,
        ema_decay: float,
        ema_eps: float,
    ) -> Self:
        """
        Create a new state instance.

        Returns
        -------
        self : DiscoValueState
            New state instance
        """
        opt_state = optim.init(params)

        return cls(
            params=params,
            optim=optim,
            opt_state=opt_state,
            adv_ema=MovingAverage(
                decay=ema_decay,
                eps=ema_eps,
            ),
            td_ema=MovingAverage(
                decay=ema_decay,
                eps=ema_eps,
            ),
        )


@struct.dataclass
class TrainerOptimizers:
    """
    Dataclass for `AgentTrainer` optimizers.

    Parameters
    ----------
    agent : optax.GradientTransformation
        Optimizer for policy agent network
    value : optax.GradientTransformation
        Optimizer for value function network
    """

    agent: optax.GradientTransformation
    value: optax.GradientTransformation


@struct.dataclass
class AgentTrainerState:
    """
    Dataclass for `AgentTrainer` state.

    Make a new instance using the `AgentTrainerState.create()` method.

    Parameters
    ----------
    params : ParameterState
        Object container policy parameter states
    optim : optax.OptState
        Policy agent optimizer state
    hidden : BundledHiddenStates
        Hidden states for recurrent policy/target networks
    current_obs : chex.Array
        Current observation from environment
    steps_trained : int (optional)
        Total training steps completed. Default is `0`
    """

    params: ParameterState
    optim: optax.GradientTransformation
    opt_state: optax.OptState

    hidden: BundledHiddenStates

    current_obs: chex.Array
    steps_trained: int = 0

    @classmethod
    def create(
        cls,
        policy_params: nnx.State,
        target_params: nnx.State,
        optim: optax.GradientTransformation,
        current_obs: chex.Array,
    ) -> Self:
        """
        Creates a new state instance.

        Parameters
        ----------
        policy_params : nnx.State
            Policy agent parameter state
        target_params : nnx.State
            Target agent parameter state
        optim : optax.GradientTransformation
            Policy agent optimizer
        current_obs : chex.Array
            Current observation

        Returns
        -------
        self : Self
            A new state instance
        """
        opt_state = optim.init(policy_params)

        return cls(
            params=ParameterState(
                policy=policy_params,
                target=target_params,
            ),
            optim=optim,
            opt_state=opt_state,
            hidden=BundledHiddenStates(),
            current_obs=current_obs,
        )

    def update_policy_params(
        self,
        new_params: nnx.State,
        new_optim_state: optax.OptState,
        tau: float,
    ) -> None:
        """
        Update policy parameters and optimizer state.

        Also updates target network via EMA (Polyak averaging):
        `target = tau * target + (1 - tau) * policy`

        Parameters
        ----------
        new_params : nnx.State
            New policy parameters
        new_optim_state : optax.OptState
            New agent optimizer state
        tau : float
            Target network update rate (Polyak averaging)
        """
        # Update target network via EMA (Polyak averaging)
        new_target_params = jax.tree.map(
            lambda old, new: tau * old + (1.0 - tau) * new,
            self.params.target,
            new_params,
        )

        new_param_state = self.params.__replace__(
            policy=new_params,
            target=new_target_params,
        )

        self = self.__replace__(
            params=new_param_state,
            opt_state=new_optim_state,
            steps_trained=self.steps_trained + 1,
        )

    def update_hidden_states(self, new_hidden: BundledHiddenStates) -> None:
        """
        Update recurrent hidden states.

        Parameters
        ----------
        new_hidden : BundledHiddenStates
            New hidden states
        """
        self = self.__replace__(hidden=new_hidden)

    def update_obs(self, new_obs: chex.Array) -> None:
        """
        Update observation state.

        Parameters
        ----------
        new_obs : chex.Array
            New current observation

        Returns
        -------
        state : AgentTrainerState
            New state with updated observation state
        """
        self = self.__replace__(current_obs=new_obs)
