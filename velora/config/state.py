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

from velora.config.outputs import AgentOutput, BufferSamples


@struct.dataclass(frozen=True)
class AgentTrainerState:
    """
    Dataclass for the `AgentTrainer` state.

    Used during checkpointing.

    Parameters
    ----------
    policy_params : chex.ArrayTree
        `PolicyAgent` network parameters
    target_params : chex.ArrayTree
        Target network parameters (EMA of policy)
    opt_state : optax.OptState
        Optimizer state for `PolicyAgent`
    h_state : chex.ArrayTree
        Hidden states for `PolicyAgent`
    buffer_state : dict
        Buffer contents and metadata
    obs : chex.Array
        Current observation from environment
    steps_trained : int
        Total training steps completed
    """

    policy_params: chex.ArrayTree
    target_params: chex.ArrayTree
    opt_state: optax.OptState
    h_state: chex.ArrayTree
    buffer_state: dict
    obs: chex.Array
    steps_trained: int


@struct.dataclass
class AgentHiddenStates:
    """
    Dataclass for agent hidden states.

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
    Dataclass for bundled hidden states used during data collection.

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
        state : Self
            An updated state
        """
        if not jnp.any(discounts == 0.0):
            return self

        def _end_check(a1: chex.Array | None, a2: chex.Array) -> chex.Array | None:
            return a1 * a2[:, None] if a1 is not None else None  # type: ignore

        discounts_expanded = jnp.repeat(discounts, n_actions)
        return self.__replace__(
            ocm=_end_check(self.ocm, discounts),
            acm=_end_check(self.acm, discounts_expanded),
            target_ocm=_end_check(self.target_ocm, discounts),
            target_acm=_end_check(self.target_acm, discounts_expanded),
        )

    def update(
        self,
        h_state: AgentHiddenStates,
        target_h_state: AgentHiddenStates,
    ) -> Self:
        """
        Update hidden states from forward pass outputs.

        Parameters
        ----------
        h_state : AgentHiddenStates
            Policy network hidden states
        target_h_state : AgentHiddenStates
            Target network hidden states

        Returns
        -------
        state : Self
            An updated state
        """
        return self.__replace__(
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
    preds: Tuple[AgentOutput, ...] = ()
    target_preds: Tuple[AgentOutput, ...] = ()
    completed_returns: Tuple[float, ...] = ()
    completed_lengths: Tuple[int, ...] = ()

    def append_step(
        self,
        actions: chex.Array,
        rewards: chex.Array,
        discounts: chex.Array,
        preds: AgentOutput,
        target_preds: AgentOutput,
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
        Stack collected data into batched arrays.

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
