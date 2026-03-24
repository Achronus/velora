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

from typing import TYPE_CHECKING, Callable, Dict, List

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

from velora.disco.inputs import GradChunkInputs
from velora.disco.outputs import HiddenShapeCache, MetaGradOutput
from velora.disco.rollouts import PoolRolloutBuffer
from velora.gym.workers import EnvWorkerPool
from velora.tracking.episode import EpisodeTracker
from velora.utils.transforms import stack_pytrees

if TYPE_CHECKING:
    from velora.disco.train import AgentTrainer


class TrainerPool:
    """
    Manages multiple agent trainer states as permanently stacked arrays.

    Contains a single pool where parameters, optimizer states, hidden states,
    and rollout buffers live as pre-stacked arrays on accelerator.

    The pool provides two main interfaces:

    - **Collection** — `collect()` runs the batched forward pass loop,
    writing directly into `PoolRolloutBuffer` instances. One vmapped
    accelerator call per step across all trainers. Environment stepping
    is delegated to an `EnvWorkerPool` of long-lived subprocesses for
    parallel execution.

    - **Gradient** — `get_grad_inputs()` slices the stacked arrays for
    a chunk of trainers. `update_from_grad()` writes gradient results
    back into the stacked arrays in-place.

    Parameters
    ----------
    trainers : List[AgentTrainer]
        Initialized trainers to pool. The pool extracts their params,
        states, envs, and episode trackers. After construction, the
        original `AgentTrainer` objects are no longer needed for
        collection or gradient computation (but are kept for metadata)
    max_actions : int
        Maximum action count across all environments
    action_masks : Dict[int, chex.Array]
        Pre-computed action masks keyed by `n_actions`
    hidden_shapes : HiddenShapeCache
        Cached hidden state shapes for zero-initialization
    n_updates : int
        Number of training rollouts per collection phase
    seq_len : int
        Timesteps per rollout
    batch_size : int
        Number of vectorized environments per trainer
    encoding_dim : int
        CNN encoder output dimensionality
    prediction_dim : int
        Prediction vector size (`y` and `z`)
    q_dim : int
        Distributional Q-value bin count
    use_bfloat16 : bool (optional)
        Cast rollout floats to `bfloat16` on accelerator transfer.
        Default is `True`
    env_worker_pool : EnvWorkerPool
        Subprocess worker pool for parallel environment stepping
    """

    def __init__(
        self,
        trainers: List["AgentTrainer"],
        max_actions: int,
        action_masks: Dict[int, chex.Array],
        hidden_shapes: HiddenShapeCache,
        n_updates: int,
        seq_len: int,
        batch_size: int,
        encoding_dim: int,
        prediction_dim: int,
        q_dim: int,
        env_worker_pool: EnvWorkerPool,
        *,
        use_bfloat16: bool = True,
    ) -> None:
        self.num_trainers = len(trainers)
        self.max_actions = max_actions

        self._hidden_shapes = hidden_shapes
        self._env_workers = env_worker_pool

        # Per-trainer metadata (not stacked)
        self.episode_trackers: List[EpisodeTracker] = [
            t.episode_tracker for t in trainers
        ]
        self.n_actions_list: List[int] = [t.n_actions for t in trainers]
        self.env_names: List[str] = [t.env_name for t in trainers]

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

        # Current observations — numpy (P, B, H, W, C)
        self.obs = self._env_workers.initial_obs.copy()

        # Rollout buffers — numpy (P, N, B, T, ...)
        self.train_buffer = PoolRolloutBuffer(
            num_trainers=self.num_trainers,
            n_rollouts=n_updates,
            n_envs=batch_size,
            seq_len=seq_len,
            n_actions=max_actions,
            encoding_dim=encoding_dim,
            prediction_dim=prediction_dim,
            q_dim=q_dim,
            use_bfloat16=use_bfloat16,
        )
        self.valid_buffer = PoolRolloutBuffer(
            num_trainers=self.num_trainers,
            n_rollouts=1,
            n_envs=batch_size,
            seq_len=seq_len * 2,
            n_actions=max_actions,
            encoding_dim=encoding_dim,
            prediction_dim=prediction_dim,
            q_dim=q_dim,
            use_bfloat16=use_bfloat16,
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
        Run one collection phase using batched accelerator forward passes.

        Each step: one vmapped accelerator call for all trainers, then
        sequential env stepping and vectorized buffer writes.

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
        P = self.num_trainers

        for _ in range(n_rollouts):
            for _ in range(seq_len):
                obs_jax = jnp.asarray(self.obs)

                # ONE vmapped accelerator call for all trainers
                (
                    preds,
                    target_preds,
                    values,
                    self.p_ocm_h,
                    self.p_acm_h,
                    self.t_ocm_h,
                    self.t_acm_h,
                    self.v_h,
                ) = batched_forward_fn(
                    obs_jax,
                    self.p_params,
                    self.t_params,
                    self.v_params,
                    self.p_ocm_h,
                    self.p_acm_h,
                    self.t_ocm_h,
                    self.t_acm_h,
                    self.v_h,
                    self.action_masks,
                )

                # Transfer to CPU in one sync
                preds_cpu, target_preds_cpu, values_cpu = jax.device_get(
                    (preds, target_preds, values)
                )
                values_np = np.asarray(values_cpu)

                # Vectorised action sampling across all trainers
                all_pi = np.asarray(preds_cpu.pi)
                if all_pi.ndim == 4:
                    all_pi = all_pi.squeeze(axis=2)

                g = np.random.gumbel(size=all_pi.shape)
                all_actions: np.ndarray = (
                    (all_pi + g).argmax(axis=-1).reshape(P, -1, 1).astype(np.int32)
                )

                # Parallel env stepping via subprocess workers
                next_obs, rewards, terminated, truncated = self._env_workers.step_all(
                    all_actions
                )

                np.copyto(self.obs, next_obs)

                all_rewards = rewards[:, :, np.newaxis].astype(np.float32)
                all_discounts = np.where(
                    terminated | truncated,
                    np.float32(0.0),
                    np.float32(1.0),
                )[:, :, np.newaxis]

                if training:
                    for i in range(P):
                        self.episode_trackers[i].record(
                            rewards[i],
                            terminated[i],
                            truncated[i],
                        )

                # Batched buffer write
                buffer.write_step_batched(
                    all_actions,
                    all_rewards,
                    all_discounts,
                    values_np,
                    preds_cpu,
                    target_preds_cpu,
                )

                # Reset hidden states on episode boundaries (vectorized accelerator)
                mask = jnp.asarray(all_discounts).squeeze(-1)  # (P, B)
                self.p_ocm_h = self.p_ocm_h * mask[..., None]
                self.p_acm_h = self.p_acm_h * mask[..., None]
                self.t_ocm_h = self.t_ocm_h * mask[..., None]
                self.t_acm_h = self.t_acm_h * mask[..., None]
                self.v_h = self.v_h * mask[..., None]

            buffer.next_rollout()

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
        disco_h: chex.Array,
        meta_h: chex.Array,
    ) -> GradChunkInputs:
        """
        Slice stacked arrays for a gradient chunk — zero copy on accelerator.

        Parameters
        ----------
        start : int
            First trainer index (inclusive)
        end : int
            Last trainer index (exclusive)
        disco_h : chex.Array
            Stacked disco hidden states for this chunk `(C, B, H)`
        meta_h : chex.Array
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
            action_masks=self.action_masks[s],
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

        # Update env
        new_obs = self._env_workers.reset_trainer(idx)
        self.obs[idx] = new_obs

        # Close main-process envs - worker owns the stepping envs
        new_trainer.envs.close()

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

    def close(self) -> None:
        """Close all environments and worker pool."""
        self._env_workers.close()
