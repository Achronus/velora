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

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import chex
import envrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from velora.disco.buffer import MixedBuffer
from velora.disco.inputs import GradChunkInputs
from velora.disco.outputs import HiddenShapeCache, MetaGradOutput
from velora.disco.rollouts import Rollout
from velora.tracking.episode import EpisodeTracker
from velora.utils.transforms import squeeze_time, stack_pytrees

if TYPE_CHECKING:
    from velora.disco.train import AgentTrainer


@struct.dataclass
class CollectCarry:
    """
    Scan carry for `TrainerPool._compiled_collect`.

    Bundles all per-step mutable state that flows through `jax.lax.scan`:
    the padded observation buffer, env-state dict, previous-action
    buffer, the five hidden-state arrays, RNG, the buffer write cursor,
    and the rollout buffer arrays themselves (as a buffer-shaped
    `Rollout`). Stacked params and action masks are passed alongside
    the carry rather than included in it.

    All per-step tensors carry a unit `B=1` axis. Collection runs one
    environment per agent (no internal vectorization), but the network
    forward expects a batch axis — so the carry preserves it. The axis
    is squeezed off at the env-step / buffer-write boundaries.

    Parameters
    ----------
    obs : jax.Array
        Padded observation buffer `(P, 1, max_obs_dim)`
    env_states : Dict[str, envrax.EnvState]
        Per-trainer env state pytrees keyed by `multi_env.env_keys`
    prev_actions : jax.Array
        Previous actions `(P, 1, max_action_dim)`
    p_ocm_h, p_acm_h, t_ocm_h, t_acm_h, v_h : jax.Array
        Stacked hidden states `(P, 1, H)`
    rng : chex.PRNGKey
        Persistent RNG, split each step
    write_idx : jax.Array
        `jnp.int32` scalar — monotonic write cursor for the rollout
        buffer. `rollout_idx = write_idx // seq_len`,
        `step_idx = write_idx % seq_len`.
    buf : Rollout
        Buffer-shaped `Rollout` `(P, N, T, ...)` flowing through
        the carry. Written via `Rollout.write_step` each step.
    """

    obs: jax.Array
    env_states: Dict[str, envrax.EnvState]
    prev_actions: jax.Array
    p_ocm_h: jax.Array
    p_acm_h: jax.Array
    t_ocm_h: jax.Array
    t_acm_h: jax.Array
    v_h: jax.Array
    rng: chex.PRNGKey
    write_idx: jax.Array
    buf: Rollout


class TrainerPool:
    """
    Manages multiple continuous-action agent trainer states as permanently
    stacked arrays on accelerator.

    Holds an `envrax.MultiEnv` so environment stepping uses same device
    as the network using JAX — no subprocess RPC or CPU/GPU bouncing.

    The pool follows the **Anakin architecture** from DeepMind's [Podracer architectures for scalable Reinforcement Learning (2021)](https://arxiv.org/abs/2104.06272)
    paper where the entire collection phase is compiled into a single
    `jax.lax.scan` (`_compiled_collect`).

    The pool provides two main interfaces:

    - **Collection** — `collect()` builds a `CollectCarry` from current
      pool state, runs `_compiled_collect` as a single fused scan, and
      writes the final carry back into the pool. Per-step rewards/dones
      come back as scan outputs for one-shot episode tracking.
    - **Gradient** — `get_grad_inputs()` slices the stacked arrays for
      a chunk of trainers. `update_from_grad()` writes gradient results
      back into the stacked arrays in-place.

    Parameters
    ----------
    trainers : List[AgentTrainer]
        Initialized trainers to pool
    max_action_dim : int
        Maximum continuous action dimensionality across all environments
    action_masks : Dict[int, jax.Array]
        Pre-computed action dimension masks keyed by `action_dim`
    hidden_shapes : HiddenShapeCache
        Cached hidden state shapes for zero-initialization
    seq_len : int
        Timesteps per rollout
    encoding_dim : int
        Encoder output dimensionality
    prediction_dim : int
        Prediction vector size (`y` and `z`)
    max_obs_dim : int
        Maximum observation dimensionality across all environments.
        Used to size the padded observation buffer.
    multi_env : envrax.MultiEnv
        Held envrax multi-env — one single `JaxEnv` per trainer.
    replay_capacity : int
        Per-agent ring size for the `MixedBuffer`.
    replay_ratio : float
        Fraction of each per-update training batch drawn from replay.
    init_rng : chex.PRNGKey
        Key used to seed `multi_env.reset` and per-step action sampling.
    """

    def __init__(
        self,
        trainers: List["AgentTrainer"],
        max_action_dim: int,
        action_masks: Dict[int, jax.Array],
        hidden_shapes: HiddenShapeCache,
        seq_len: int,
        encoding_dim: int,
        prediction_dim: int,
        max_obs_dim: int,
        multi_env: envrax.MultiEnv,
        replay_capacity: int,
        replay_ratio: float,
        init_rng: chex.PRNGKey,
    ) -> None:
        self.num_trainers = len(trainers)
        self.max_obs_dim = max_obs_dim

        self._hidden_shapes = hidden_shapes
        self.multi_env = multi_env

        # Per-trainer dict keys into `multi_env.envs`
        self._env_keys: List[str] = multi_env.env_keys

        # Per-trainer metadata (not stacked)
        self.episode_trackers: List[EpisodeTracker] = [
            t.episode_tracker for t in trainers
        ]
        self.n_actions_list: List[int] = [t.n_actions for t in trainers]
        self.env_names: List[str] = [t.env_name for t in trainers]

        # Per-trainer true (unpadded) obs / action dims — for pack/unpack.
        # Dict from envrax flattened to list in trainer order.
        obs_sizes = multi_env.observation_sizes
        action_sizes = multi_env.action_sizes
        self._obs_dims: List[int] = [obs_sizes[k] for k in self._env_keys]
        self._action_dims: List[int] = [action_sizes[k] for k in self._env_keys]

        # Lifetime tracking
        self._env_steps: List[int] = [t._env_steps for t in trainers]
        self._step_budgets: List[int] = [t._step_budget for t in trainers]
        self._collection_steps: List[int] = [t._collection_step for t in trainers]

        # Shared meta-gradient optimizer — one jitted update, vmapped over stacked state
        shared_meta_optim = trainers[0].meta_optim
        self.meta_optim_update: Callable = (
            jax.jit(shared_meta_optim.update)
            if trainers[0].jit_compile
            else shared_meta_optim.update
        )
        self.meta_opt_state = stack_pytrees([t.meta_opt_state for t in trainers])

        # Stacked parameters on accelerator (P, ...)
        self.p_params = stack_pytrees([t.policy_agent.get_params() for t in trainers])
        self.t_params = stack_pytrees([t.target_agent.get_params() for t in trainers])
        self.v_params = stack_pytrees([t.value_agent.get_params() for t in trainers])

        # Stacked optimizer states on accelerator (P, ...)
        self.p_opt = stack_pytrees([t.state.policy_opt_state for t in trainers])
        self.v_opt = stack_pytrees([t.state.value_opt_state for t in trainers])

        # Stacked EMA states on accelerator (P, ...)
        self.adv_ema = stack_pytrees([t.state.adv_ema for t in trainers])
        self.td_ema = stack_pytrees([t.state.td_ema for t in trainers])

        # Stacked action masks on accelerator (P, max_actions)
        self.action_masks = jnp.stack([action_masks[t.n_actions] for t in trainers])

        # Collection hidden states on accelerator (P, H)
        self._init_hidden(trainers)

        # Persistent RNG used for env resets / action sampling
        self._rng = init_rng

        # Reset all envs and pack into padded (P, max_obs_dim) buffer
        self._rng, reset_rng = jax.random.split(self._rng)
        obs_dict, env_states = multi_env.reset(reset_rng)
        self.env_states = env_states
        self.obs = self._pack_obs(obs_dict)

        # Rollout buffers — on device
        buf_kwargs: Dict[str, Any] = dict(
            n_actions=max_action_dim,
            encoding_dim=encoding_dim,
            prediction_dim=prediction_dim,
        )

        # Training: fresh+replay mixed ring, sampled per agent update
        self.train_buffer = MixedBuffer.create(
            num_trainers=self.num_trainers,
            capacity=replay_capacity,
            replay_ratio=replay_ratio,
            seq_len=seq_len,
            **buf_kwargs,
        )

        # Validation: raw `Rollout` of shape (P, 1, 2T, ...)
        self.valid_rollout: Rollout = Rollout.zeros(
            shape_prefix=(self.num_trainers, 1, seq_len * 2),
            **buf_kwargs,
        )

        # Track previous actions for action-conditional forward passes (P, 1, max_action_dim)
        self.prev_actions = jnp.zeros(
            (self.num_trainers, 1, max_action_dim),
            dtype=jnp.float32,
        )

        # Build the JIT-cached collect kernel
        self._compiled_collect = self._build_collect_kernel()

    def _build_collect_kernel(self) -> Callable:
        """
        Compile the collection scan once, return the cached kernel.

        The kernel closes over `self` so it has access to `multi_env`,
        `_pack_obs`, `_unpack_actions`, and `_env_keys` without making
        them traced arguments. Static args (`batched_forward_fn`,
        `total_steps`, `seq_len`) gate the JIT cache so the training
        (`n_updates * seq_len`) and validation (`2 * seq_len`) variants
        each compile to their own program once and are then reused.
        """
        multi_env = self.multi_env
        pack_obs = self._pack_obs
        unpack_actions = self._unpack_actions
        env_keys = self._env_keys

        @partial(
            jax.jit,
            static_argnames=("batched_forward_fn", "total_steps", "seq_len"),
        )
        def collect_kernel(
            batched_forward_fn: Callable,
            init_carry: CollectCarry,
            total_steps: int,
            seq_len: int,
            slot_indices: jax.Array,
            p_params: chex.ArrayTree,
            t_params: chex.ArrayTree,
            v_params: chex.ArrayTree,
            action_masks: jax.Array,
        ) -> Tuple[CollectCarry, jax.Array, jax.Array]:
            def _step_body(
                carry: CollectCarry, _
            ) -> Tuple[CollectCarry, Tuple[jax.Array, jax.Array]]:
                (
                    preds,
                    target_preds,
                    values,
                    new_p_ocm_h,
                    new_p_acm_h,
                    new_t_ocm_h,
                    new_t_acm_h,
                    new_v_h,
                ) = batched_forward_fn(
                    carry.obs,
                    carry.prev_actions,
                    p_params,
                    t_params,
                    v_params,
                    carry.p_ocm_h,
                    carry.p_acm_h,
                    carry.t_ocm_h,
                    carry.t_acm_h,
                    carry.v_h,
                    action_masks,
                )

                rng, noise_rng = jax.random.split(carry.rng)
                mu = preds.mu
                log_std = preds.log_std
                std = jnp.exp(log_std)
                noise = jax.random.normal(noise_rng, mu.shape)
                actions = (mu + std * noise).astype(jnp.float32)

                actions_per_group = unpack_actions(actions)
                obs_dict, new_env_states, rewards_dict, dones_dict, _ = multi_env.step(
                    carry.env_states, actions_per_group
                )
                new_obs = pack_obs(obs_dict)
                rewards = jnp.stack([rewards_dict[k] for k in env_keys], axis=0)[
                    ..., None
                ]
                dones = jnp.stack([dones_dict[k] for k in env_keys], axis=0)
                discounts = jnp.where(dones[:, None], 0.0, 1.0)

                rollout_idx = carry.write_idx // seq_len
                step_idx = carry.write_idx % seq_len
                slot_idx = slot_indices[rollout_idx]
                new_buf = carry.buf.write_step(
                    slot_idx,
                    step_idx,
                    squeeze_time(actions),
                    rewards,
                    discounts,
                    squeeze_time(values),
                    jax.tree.map(squeeze_time, preds),
                    jax.tree.map(squeeze_time, target_preds),
                )

                mask = discounts[:, :, None]

                new_carry = CollectCarry(
                    obs=new_obs,
                    env_states=new_env_states,
                    prev_actions=actions * mask,
                    p_ocm_h=new_p_ocm_h * mask,
                    p_acm_h=new_p_acm_h * mask,
                    t_ocm_h=new_t_ocm_h * mask,
                    t_acm_h=new_t_acm_h * mask,
                    v_h=new_v_h * mask,
                    rng=rng,
                    write_idx=carry.write_idx + 1,
                    buf=new_buf,
                )
                return new_carry, (rewards.squeeze(-1), dones)

            final_carry, (all_rewards, all_dones) = jax.lax.scan(
                _step_body,
                init_carry,
                None,
                length=total_steps,
            )
            return final_carry, all_rewards, all_dones

        return collect_kernel

    def _init_hidden(self, trainers: List["AgentTrainer"]) -> None:
        """
        Stack collection hidden states from trainers, replacing `None`
        with zeros.
        """
        s = self._hidden_shapes

        def _h_or_zeros(h, shape):
            return h if h is not None else jnp.zeros(shape)

        self.p_ocm_h = jnp.stack(
            [_h_or_zeros(t.state.hidden.policy_ocm, s.policy_ocm) for t in trainers]
        )
        self.p_acm_h = jnp.stack(
            [_h_or_zeros(t.state.hidden.policy_acm, s.policy_acm) for t in trainers]
        )
        self.t_ocm_h = jnp.stack(
            [_h_or_zeros(t.state.hidden.target_ocm, s.target_ocm) for t in trainers]
        )
        self.t_acm_h = jnp.stack(
            [_h_or_zeros(t.state.hidden.target_acm, s.target_acm) for t in trainers]
        )
        self.v_h = jnp.stack(
            [_h_or_zeros(t.state.hidden.value, s.value) for t in trainers]
        )

    def _pack_obs(self, obs_dict: Dict[str, jax.Array]) -> jax.Array:
        """
        Zero-pad per-env observations `(obs_i,)` to `(max_obs_dim,)`,
        stack into `(P, max_obs_dim)`, and add a unit `B=1` axis to
        match the carry contract: `(P, 1, max_obs_dim)`.
        """
        padded = [
            jnp.pad(obs_dict[k], (0, self.max_obs_dim - self._obs_dims[i]))
            for i, k in enumerate(self._env_keys)
        ]
        return jnp.stack(padded, axis=0)[:, None, :]

    def _unpack_actions(self, actions: jax.Array) -> Dict[str, jax.Array]:
        """
        Slice the padded action array `(P, 1, max_action_dim)` into a
        dict of per-env `(action_i,)` arrays for `MultiEnv.step`. The
        unit `B=1` axis is dropped at slot 1.
        """
        return {
            k: actions[i, 0, : self._action_dims[i]]
            for i, k in enumerate(self._env_keys)
        }

    def needs_reset(self, idx: int) -> bool:
        """Check if trainer at `idx` has exhausted its lifetime budget."""
        return self._env_steps[idx] >= self._step_budgets[idx]

    def collect(
        self,
        batched_forward_fn: Callable,
        n_rollouts: int,
        seq_len: int,
        training: bool,
    ) -> None:
        """
        Run one collection phase as a single compiled `jax.lax.scan`.

        Parameters
        ----------
        batched_forward_fn : Callable
            JIT + vmap compiled forward function
        n_rollouts : int
            Number of rollouts to collect
        seq_len : int
            Timesteps per rollout
        training : bool
            Whether this is training (writes into `train_buffer`'s ring)
            or validation (overwrites `valid_rollout`).
        """
        total_steps = n_rollouts * seq_len

        start_slot = self.train_buffer.write_idx
        if training:
            slot_indices = (
                start_slot + jnp.arange(n_rollouts, dtype=jnp.int32)
            ) % self.train_buffer.capacity
            init_buf = self.train_buffer.storage
        else:
            slot_indices = jnp.arange(n_rollouts, dtype=jnp.int32)
            init_buf = self.valid_rollout

        self._rng, scan_rng = jax.random.split(self._rng)
        init_carry = CollectCarry(
            obs=self.obs,
            env_states=self.env_states,
            prev_actions=self.prev_actions,
            p_ocm_h=self.p_ocm_h,
            p_acm_h=self.p_acm_h,
            t_ocm_h=self.t_ocm_h,
            t_acm_h=self.t_acm_h,
            v_h=self.v_h,
            rng=scan_rng,
            write_idx=jnp.int32(0),
            buf=init_buf,
        )

        final_carry, all_rewards, all_dones = self._compiled_collect(
            batched_forward_fn,
            init_carry,
            total_steps,
            seq_len,
            slot_indices,
            self.p_params,
            self.t_params,
            self.v_params,
            self.action_masks,
        )

        # Write carry state back into the pool
        self.obs = final_carry.obs
        self.env_states = final_carry.env_states
        self.prev_actions = final_carry.prev_actions
        self.p_ocm_h = final_carry.p_ocm_h
        self.p_acm_h = final_carry.p_acm_h
        self.t_ocm_h = final_carry.t_ocm_h
        self.t_acm_h = final_carry.t_acm_h
        self.v_h = final_carry.v_h
        self._rng = final_carry.rng

        if training:
            new_write_idx = jnp.asarray(
                (start_slot + n_rollouts) % self.train_buffer.capacity,
                dtype=jnp.int32,
            )
            self.train_buffer = self.train_buffer.__replace__(
                storage=final_carry.buf,
                write_idx=new_write_idx,
                valid_count=jnp.minimum(
                    self.train_buffer.valid_count + n_rollouts,
                    self.train_buffer.capacity,
                ),
            )

            # Episode tracking: one device_get for the whole rollout
            rewards_cpu = np.asarray(jax.device_get(all_rewards))  # (T_total, P)
            dones_cpu = np.asarray(jax.device_get(all_dones))  # (T_total, P)
            t_total = rewards_cpu.shape[0]

            for i in range(self.num_trainers):
                tracker = self.episode_trackers[i]
                for t in range(t_total):
                    # tracker expects (num_envs=1,) shaped arrays
                    tracker.record(
                        rewards_cpu[t, i : i + 1],
                        dones_cpu[t, i : i + 1],
                    )
        else:
            self.valid_rollout = final_carry.buf

    def soft_update_targets(self, tau: float) -> None:
        """
        Vectorized soft parameter update for all target agents.

        Formula: `θ_target ← τ * θ_policy + (1 - τ) * θ_target`

        Parameters
        ----------
        tau : float
            Soft update coefficient
        """
        self.t_params = jax.tree.map(
            lambda p, t: tau * p + (1.0 - tau) * t,
            self.p_params,
            self.t_params,
        )

    def finalize_training(self, n_rollouts: int, seq_len: int) -> None:
        """
        Post-training-phase bookkeeping: update step budgets and
        collection step counters.

        Parameters
        ----------
        n_rollouts : int
            Number of trajectories collected this meta-step (matches the
            `n_rollouts` passed to the corresponding `collect` call).
        seq_len : int
            Timesteps per rollout
        """
        steps = n_rollouts * seq_len

        for i in range(self.num_trainers):
            self._env_steps[i] += steps
            self._collection_steps[i] += 1

    def get_grad_inputs(
        self,
        disco_h: jax.Array,
        meta_h: jax.Array,
        rng: chex.PRNGKey,
        n_updates: int,
        batch_size: int,
    ) -> GradChunkInputs:
        """
        Bundle the full-population gradient inputs — zero copy on accelerator.

        Parameters
        ----------
        disco_h : jax.Array
            Stacked disco hidden states `(P, B, H)`
        meta_h : jax.Array
            Stacked meta hidden states `(P, B, H)`
        rng : chex.PRNGKey
            RNG used to draw replay samples from `train_buffer`.
        n_updates : int
            Number of per-update batches to produce (`N`).
        batch_size : int
            Trajectories per per-update batch (`B`).

        Returns
        -------
        inputs : GradChunkInputs
            All inputs needed for `_batch_grad_fn` (excluding
            `meta_params` which is shared).
        """
        return GradChunkInputs(
            p_params=self.p_params,
            v_params=self.v_params,
            disco_h=disco_h,
            meta_h=meta_h,
            p_opt=self.p_opt,
            v_opt=self.v_opt,
            adv_ema=self.adv_ema,  # type: ignore
            td_ema=self.td_ema,  # type: ignore
            train_rollout=self.train_buffer.sample(rng, n_updates, batch_size),
            valid_rollout=self.valid_rollout,
            masks=self.action_masks,
        )

    def update_from_grad(self, chunk_out: MetaGradOutput) -> None:
        """
        Replace the pool's stacked state with the gradient-fn outputs.

        Direct attribute reassignment — no scatter kernel needed since
        every leaf is full-`P`-shaped and replaces the prior tensor
        outright. Target params hard-sync to the freshly-updated policy
        params (existing behavior; the subsequent `soft_update_targets`
        call lerps from there).

        Parameters
        ----------
        chunk_out : MetaGradOutput
            Output from `_batch_grad_fn`.
        """
        self.p_params = chunk_out.p_params
        self.v_params = chunk_out.v_params
        self.t_params = chunk_out.p_params
        self.v_opt = chunk_out.v_opt_state
        self.adv_ema = chunk_out.adv_ema
        self.td_ema = chunk_out.td_ema

    def reset_trainer(
        self,
        idx: int,
        new_trainer: "AgentTrainer",
    ) -> None:
        """
        Replace a trainer at `idx` with fresh params and state.

        Writes new parameters, optimizer states, and hidden states into
        the stacked arrays at the given index using in-place updates.

        Parameters
        ----------
        idx : int
            Trainer index to reset
        new_trainer : AgentTrainer
            Freshly initialized trainer
        """
        # Write fresh params into stacked arrays
        new_p = new_trainer.policy_agent.get_params()
        new_t = new_trainer.target_agent.get_params()
        new_v = new_trainer.value_agent.get_params()

        self.p_params = jax.tree.map(
            lambda dst, src: dst.at[idx].set(src),
            self.p_params,
            new_p,
        )
        self.t_params = jax.tree.map(
            lambda dst, src: dst.at[idx].set(src),
            self.t_params,
            new_t,
        )
        self.v_params = jax.tree.map(
            lambda dst, src: dst.at[idx].set(src),
            self.v_params,
            new_v,
        )

        # Fresh optimizer states
        self.p_opt = jax.tree.map(
            lambda dst, src: dst.at[idx].set(src),
            self.p_opt,
            new_trainer.state.policy_opt_state,
        )
        self.v_opt = jax.tree.map(
            lambda dst, src: dst.at[idx].set(src),
            self.v_opt,
            new_trainer.state.value_opt_state,
        )

        # Reset EMA states
        self.adv_ema = jax.tree.map(
            lambda dst, src: dst.at[idx].set(src),
            self.adv_ema,
            new_trainer.state.adv_ema,
        )
        self.td_ema = jax.tree.map(
            lambda dst, src: dst.at[idx].set(src),
            self.td_ema,
            new_trainer.state.td_ema,
        )

        # Clear this trainer's replay slice so the new agent doesn't
        # sample stale trajectories from the previous one
        self.train_buffer = self.train_buffer.reset_at(idx)

        # Zero hidden states at index
        s = self._hidden_shapes
        self.p_ocm_h = self.p_ocm_h.at[idx].set(jnp.zeros(s.policy_ocm))
        self.p_acm_h = self.p_acm_h.at[idx].set(jnp.zeros(s.policy_acm))
        self.t_ocm_h = self.t_ocm_h.at[idx].set(jnp.zeros(s.target_ocm))
        self.t_acm_h = self.t_acm_h.at[idx].set(jnp.zeros(s.target_acm))
        self.v_h = self.v_h.at[idx].set(jnp.zeros(s.value))

        # Reset env for this trainer — dict-keyed access via env_keys
        self._rng, reset_rng = jax.random.split(self._rng)
        key = self._env_keys[idx]
        new_obs, new_env_state = self.multi_env.envs[key].reset(reset_rng)
        self.env_states = {**self.env_states, key: new_env_state}

        padded_obs = jnp.pad(new_obs, (0, self.max_obs_dim - self._obs_dims[idx]))
        self.obs = self.obs.at[idx, 0].set(padded_obs)

        # Reset prev_actions for this trainer
        self.prev_actions = self.prev_actions.at[idx].set(0.0)

        # Update tracker, and per-trainer metadata
        self.episode_trackers[idx] = new_trainer.episode_tracker
        self.n_actions_list[idx] = new_trainer.n_actions
        self.env_names[idx] = new_trainer.env_name

        # Lifetime tracking
        self._env_steps[idx] = new_trainer._env_steps
        self._step_budgets[idx] = new_trainer._step_budget
        self._collection_steps[idx] = new_trainer._collection_step

        # Meta-gradient optimizer
        # Scatter fresh per-trainer meta-opt state into the stacked tree
        self.meta_opt_state = jax.tree.map(
            lambda dst, src: dst.at[idx].set(src),
            self.meta_opt_state,
            new_trainer.meta_opt_state,
        )
