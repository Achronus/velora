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

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import chex
import envrax
import jax
import jax.numpy as jnp
import optax
from flax import struct

from velora.disco.inputs import GradChunkInputs
from velora.disco.outputs import HiddenShapeCache, MetaGradOutput
from velora.disco.rollouts import PoolRolloutBuffer, Rollout
from velora.tracking.episode import EpisodeTracker
from velora.utils.transforms import stack_pytrees

if TYPE_CHECKING:
    from velora.disco.train import AgentTrainer


@struct.dataclass
class CollectCarry:
    """
    Scan carry for `TrainerPool._compiled_collect`.

    Bundles all per-step mutable state that flows through `jax.lax.scan`:
    the padded observation buffer, env-state tuple, previous-action
    buffer, the five hidden-state arrays, RNG, the buffer write cursor,
    and the rollout buffer arrays themselves (as a buffer-shaped
    `Rollout`). Stacked params and action masks are passed alongside
    the carry rather than included in it.

    Parameters
    ----------
    obs : jax.Array
        Padded observation buffer `(P, B, max_obs_dim)`
    env_states : Tuple[Any, ...]
        Per-trainer env state pytrees (one per `VecEnv` group)
    prev_actions : jax.Array
        Previous actions `(P, B, max_action_dim)`
    p_ocm_h, p_acm_h, t_ocm_h, t_acm_h, v_h : jax.Array
        Stacked hidden states `(P, B, H)`
    rng : chex.PRNGKey
        Persistent RNG, split each step
    write_idx : jax.Array
        `jnp.int32` scalar — monotonic write cursor for the rollout
        buffer. `rollout_idx = write_idx // seq_len`,
        `step_idx = write_idx % seq_len`.
    buf : Rollout
        Buffer-shaped `Rollout` `(P, N, B, T, ...)` flowing through
        the carry. Written via `Rollout.write_step` each step.
    """

    obs: jax.Array
    env_states: Tuple[Any, ...]
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

    Holds an `envrax.MultiVecEnv` so environment stepping uses same device
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
    n_updates : int
        Number of training rollouts per collection phase
    seq_len : int
        Timesteps per rollout
    batch_size : int
        Number of vectorized environments per trainer
    encoding_dim : int
        Encoder output dimensionality
    prediction_dim : int
        Prediction vector size (`y` and `z`)
    max_obs_dim : int
        Maximum observation dimensionality across all environments.
        Used to size the padded observation buffer.
    multi_vec_env : envrax.MultiVecEnv
        Held envrax multi-vec environment — one `VecEnv` per trainer.
    init_rng : chex.PRNGKey
        Key used to seed `multi_vec_env.reset` and per-step action sampling.
    """

    def __init__(
        self,
        trainers: List["AgentTrainer"],
        max_action_dim: int,
        action_masks: Dict[int, jax.Array],
        hidden_shapes: HiddenShapeCache,
        n_updates: int,
        seq_len: int,
        batch_size: int,
        encoding_dim: int,
        prediction_dim: int,
        max_obs_dim: int,
        multi_vec_env: envrax.MultiVecEnv,
        init_rng: chex.PRNGKey,
    ) -> None:
        self.num_trainers = len(trainers)
        self.max_obs_dim = max_obs_dim
        self.batch_size = batch_size

        self._hidden_shapes = hidden_shapes
        self.multi_vec_env = multi_vec_env

        # Per-trainer metadata (not stacked)
        self.episode_trackers: List[EpisodeTracker] = [
            t.episode_tracker for t in trainers
        ]
        self.n_actions_list: List[int] = [t.n_actions for t in trainers]
        self.env_names: List[str] = [t.env_name for t in trainers]

        # Per-trainer true (unpadded) obs / action dims — for pack/unpack
        self._obs_dims = multi_vec_env.single_observation_sizes
        self._action_dims = multi_vec_env.single_action_sizes

        # Lifetime tracking
        self._env_steps: List[int] = [t._env_steps for t in trainers]
        self._step_budgets: List[int] = [t._step_budget for t in trainers]
        self._collection_steps: List[int] = [t._collection_step for t in trainers]

        # Per-trainer meta-gradient optimizers (can't be stacked — different states)
        self.meta_optim_updates: List[Callable] = [
            t.meta_optim_update for t in trainers
        ]
        self.meta_opt_states: List[optax.OptState] = [
            t.meta_opt_state for t in trainers
        ]

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

        # Collection hidden states on accelerator (P, B, H)
        self._init_hidden(trainers)

        # Persistent RNG used for env resets / action sampling
        self._rng = init_rng

        # Reset all envs and pack into padded (P, B, max_obs_dim) buffer
        self._rng, reset_rng = jax.random.split(self._rng)
        obs_list, env_state_list = multi_vec_env.reset(reset_rng)
        self.env_states = tuple(env_state_list)
        self.obs = self._pack_obs(obs_list)  # type: ignore

        # Rollout buffers — on device (P, N, B, T, ...)
        buf_kwargs: Dict[str, Any] = dict(
            n_envs=batch_size,
            n_actions=max_action_dim,
            encoding_dim=encoding_dim,
            prediction_dim=prediction_dim,
        )

        self.train_buffer = PoolRolloutBuffer(
            num_trainers=self.num_trainers,
            n_rollouts=n_updates,
            seq_len=seq_len,
            **buf_kwargs,
        )
        self.valid_buffer = PoolRolloutBuffer(
            num_trainers=self.num_trainers,
            n_rollouts=1,
            seq_len=seq_len * 2,
            **buf_kwargs,
        )

        # Track previous actions for action-conditional forward passes (P, B, max_action_dim)
        self.prev_actions = jnp.zeros(
            (self.num_trainers, batch_size, max_action_dim),
            dtype=jnp.float32,
        )

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

    def _pack_obs(self, obs_list: List[jax.Array]) -> jax.Array:
        """
        Zero-pad per-env observations `(B, obs_i)` to `(B, max_obs_dim)`
        and stack into a `(P, B, max_obs_dim)` array.
        """
        padded = [
            jnp.pad(o, ((0, 0), (0, self.max_obs_dim - self._obs_dims[i])))
            for i, o in enumerate(obs_list)
        ]
        return jnp.stack(padded, axis=0)

    def _unpack_actions(self, actions: jax.Array) -> List[jax.Array]:
        """
        Slice the padded action array `(P, B, max_action_dim)` into a list
        of per-env `(B, action_i)` arrays for `MultiVecEnv.step`.
        """
        return [actions[i, :, : self._action_dims[i]] for i in range(self.num_trainers)]

    def needs_reset(self, idx: int) -> bool:
        """Check if trainer at `idx` has exhausted its lifetime budget."""
        return self._env_steps[idx] >= self._step_budgets[idx]

    def _compiled_collect(
        self,
        batched_forward_fn: Callable,
        init_carry: CollectCarry,
        total_steps: int,
        seq_len: int,
        p_params: chex.ArrayTree,
        t_params: chex.ArrayTree,
        v_params: chex.ArrayTree,
        action_masks: jax.Array,
    ) -> Tuple[CollectCarry, jax.Array, jax.Array]:
        """
        Run the full collection phase as one `jax.lax.scan`.

        The scan body covers forward pass, Gaussian action sampling,
        `MultiVecEnv.step` (Python for-loop traced away into fused XLA
        kernels), `_pack_obs` / `_unpack_actions`, buffer writes via
        `Rollout.write_step`, and hidden-state masking on episode
        boundaries. Per-step rewards and dones are emitted as scan
        outputs for post-scan episode tracking.

        Parameters
        ----------
        batched_forward_fn : Callable
            JIT + vmap'd forward function — inlines into the scan body
        init_carry : CollectCarry
            Initial scan carry built from current pool state
        total_steps : int
            Total inner steps `n_rollouts * seq_len`
        seq_len : int
            Per-rollout timestep count, used to derive
            `(rollout_idx, step_idx)` from `write_idx`
        p_params, t_params, v_params : chex.ArrayTree
            Stacked policy / target / value params
        action_masks : jax.Array
            Stacked per-trainer action-dim masks

        Returns
        -------
        final_carry : CollectCarry
            Final pool state after `total_steps` inner steps
        all_rewards : jax.Array
            Per-step rewards `(total_steps, P, B)` for episode tracking
        all_dones : jax.Array
            Per-step dones `(total_steps, P, B)` for episode tracking
        """
        multi_vec_env = self.multi_vec_env
        pack_obs = self._pack_obs
        unpack_actions = self._unpack_actions

        def _step_body(
            carry: CollectCarry, _
        ) -> Tuple[CollectCarry, Tuple[jax.Array, jax.Array]]:
            # 1. Forward — JIT'd fn inlines into the scan HLO
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

            # 2. Gaussian reparameterised action sample
            rng, noise_rng = jax.random.split(carry.rng)
            mu = preds.mu
            log_std = preds.log_std

            if mu.ndim == 4:
                mu = jnp.squeeze(mu, axis=2)
                log_std = jnp.squeeze(log_std, axis=2)

            std = jnp.exp(log_std)
            noise = jax.random.normal(noise_rng, mu.shape)
            actions = (mu + std * noise).astype(jnp.float32)

            # 3. Env step — for-loop inside MultiVecEnv.step traces away
            actions_per_group = unpack_actions(actions)
            obs_list, new_env_states_list, rewards_list, dones_list, _ = (
                multi_vec_env.step(
                    list(carry.env_states),
                    actions_per_group,  # type: ignore
                )
            )
            new_obs = pack_obs(obs_list)  # type: ignore
            rewards = jnp.stack(rewards_list, axis=0)[..., None]  # (P, B, 1)
            dones = jnp.stack(dones_list, axis=0)  # (P, B)
            discounts = jnp.where(dones, 0.0, 1.0)[..., None]  # (P, B, 1)

            # 4. Buffer write via JAX scalar indices
            rollout_idx = carry.write_idx // seq_len
            step_idx = carry.write_idx % seq_len
            new_buf = carry.buf.write_step(
                rollout_idx,
                step_idx,
                actions,
                rewards,
                discounts,
                values,
                preds,
                target_preds,
            )

            # 5. Mask hidden states + prev_actions on episode boundary
            mask = discounts.squeeze(-1)[..., None]  # (P, B, 1)

            new_carry = CollectCarry(
                obs=new_obs,
                env_states=tuple(new_env_states_list),
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

    def collect(
        self,
        batched_forward_fn: Callable,
        n_rollouts: int,
        seq_len: int,
        training: bool,
    ) -> None:
        """
        Run one collection phase as a single compiled `jax.lax.scan`.

        Builds an initial `CollectCarry` from current pool state, calls
        `_compiled_collect`, writes the final carry back into the pool,
        loads the buffer's post-scan arrays, and (for training) records
        per-step rewards / dones into the episode trackers with one
        `jax.device_get`.

        Parameters
        ----------
        batched_forward_fn : Callable
            JIT + vmap compiled forward function
        n_rollouts : int
            Number of rollouts to collect
        seq_len : int
            Timesteps per rollout
        training : bool
            Whether this is training (uses `train_buffer`) or
            validation (uses `valid_buffer`)
        """
        buffer = self.train_buffer if training else self.valid_buffer
        total_steps = n_rollouts * seq_len

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
            buf=buffer.as_pytree(),
        )

        final_carry, all_rewards, all_dones = self._compiled_collect(
            batched_forward_fn,
            init_carry,
            total_steps,
            seq_len,
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
        buffer.load_from_pytree(final_carry.buf)

        # Episode tracking: one device_get for the whole rollout
        if training:
            rewards_cpu: jax.Array = jax.device_get(all_rewards)  # (T_total, P, B)
            dones_cpu: jax.Array = jax.device_get(all_dones)  # (T_total, P, B)
            t_total = jnp.shape(rewards_cpu)[0]
            for i in range(self.num_trainers):
                tracker = self.episode_trackers[i]
                for t in range(t_total):
                    tracker.record(
                        rewards_cpu[t, i],  # type: ignore
                        dones_cpu[t, i],  # type: ignore
                        dones_cpu[t, i],  # type: ignore
                    )

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

    def finalize_training(self, n_updates: int, seq_len: int) -> None:
        """
        Post-training-phase bookkeeping: update step budgets and
        collection step counters.

        Parameters
        ----------
        n_updates : int
            Number of rollouts collected
        seq_len : int
            Timesteps per rollout
        """
        steps = n_updates * seq_len

        for i in range(self.num_trainers):
            self._env_steps[i] += steps
            self._collection_steps[i] += 1

    def get_grad_inputs(
        self,
        start: int,
        end: int,
        disco_h: jax.Array,
        meta_h: jax.Array,
    ) -> GradChunkInputs:
        """
        Slice stacked arrays for a gradient chunk — zero copy on accelerator.

        Parameters
        ----------
        start : int
            First trainer index (inclusive)
        end : int
            Last trainer index (exclusive)
        disco_h : jax.Array
            Stacked disco hidden states for this chunk `(C, B, H)`
        meta_h : jax.Array
            Stacked meta hidden states for this chunk `(C, B, H)`

        Returns
        -------
        inputs : GradChunkInputs
            All inputs needed for `_batch_grad_fn` (excluding
            `meta_params` which is shared)
        """
        s = slice(start, end)
        _s = lambda x: jax.tree.map(lambda a: a[s], x)  # noqa: E731

        return GradChunkInputs(
            p_params=_s(self.p_params),
            v_params=_s(self.v_params),
            disco_h=disco_h,
            meta_h=meta_h,
            p_opt=_s(self.p_opt),
            v_opt=_s(self.v_opt),
            adv_ema=_s(self.adv_ema),
            td_ema=_s(self.td_ema),
            train_rollout=self.train_buffer.get_chunk(start, end),
            valid_rollout=self.valid_buffer.get_chunk(start, end, squeeze_n=True),
            masks=self.action_masks[s],
        )

    def update_from_grad(
        self,
        start: int,
        end: int,
        chunk_out: MetaGradOutput,
    ) -> None:
        """
        Write gradient computation results back into stacked arrays.

        Uses `jax.tree.map` with `.at[start:end].set()` for
        in-place accelerator updates without creating new arrays.

        Parameters
        ----------
        start : int
            First trainer index (inclusive)
        end : int
            Last trainer index (exclusive)
        chunk_out : MetaGradOutput
            Output from `_batch_grad_fn`
        """
        s = slice(start, end)

        # Update policy and value params
        self.p_params = jax.tree.map(
            lambda dst, src: dst.at[s].set(src),
            self.p_params,
            chunk_out.p_params,
        )
        self.v_params = jax.tree.map(
            lambda dst, src: dst.at[s].set(src),
            self.v_params,
            chunk_out.v_params,
        )

        # Update target params
        self.t_params = jax.tree.map(
            lambda dst, src: dst.at[s].set(src),
            self.t_params,
            chunk_out.p_params,
        )

        # Update value optimizer state
        self.v_opt = jax.tree.map(
            lambda dst, src: dst.at[s].set(src),
            self.v_opt,
            chunk_out.v_opt_state,
        )

        # Update EMA states
        self.adv_ema = jax.tree.map(
            lambda dst, src: dst.at[s].set(src),
            self.adv_ema,
            chunk_out.adv_ema,
        )
        self.td_ema = jax.tree.map(
            lambda dst, src: dst.at[s].set(src),
            self.td_ema,
            chunk_out.td_ema,
        )

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

        # Zero hidden states at index
        s = self._hidden_shapes
        self.p_ocm_h = self.p_ocm_h.at[idx].set(jnp.zeros(s.policy_ocm))
        self.p_acm_h = self.p_acm_h.at[idx].set(jnp.zeros(s.policy_acm))
        self.t_ocm_h = self.t_ocm_h.at[idx].set(jnp.zeros(s.target_ocm))
        self.t_acm_h = self.t_acm_h.at[idx].set(jnp.zeros(s.target_acm))
        self.v_h = self.v_h.at[idx].set(jnp.zeros(s.value))

        # Reset env for this trainer
        self._rng, reset_rng = jax.random.split(self._rng)
        new_obs, new_env_state = self.multi_vec_env.reset_at(idx, reset_rng)
        self.env_states = tuple(
            new_env_state if i == idx else s for i, s in enumerate(self.env_states)
        )

        padded_obs = jnp.pad(
            new_obs, ((0, 0), (0, self.max_obs_dim - self._obs_dims[idx]))
        )
        self.obs = self.obs.at[idx].set(padded_obs)

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
        self.meta_optim_updates[idx] = new_trainer.meta_optim_update
        self.meta_opt_states[idx] = new_trainer.meta_opt_state
