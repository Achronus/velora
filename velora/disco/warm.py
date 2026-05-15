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

from typing import TYPE_CHECKING, Callable

import chex
import jax
import jax.numpy as jnp

from velora.disco.pool import CollectCarry

if TYPE_CHECKING:
    from velora.disco.train import RuleTrainer


class WarmCompile:
    """
    Pre-compile every JAX kernel exercised by the meta-training loop.

    Extracted from `RuleTrainer` to keep the warm-up surface scannable.
    The class is a one-shot helper: construct, call `run()`, discard.
    Each step ticks the trainer's dashboard so the progress bar reflects
    real work.

    Covers:
        1. Shared `_meta_optim_update` JIT
        2. `pool._compiled_collect` for the training phase
           (`n_updates * seq_len` inner steps) — one scan kernel covers
           the vmapped forward, action sampling, `MultiVecEnv.step`,
           buffer writes via `Rollout.write_step`, and hidden masks.
        3. `pool._compiled_collect` for the validation phase
           (`1 * (seq_len * 2)` inner steps) — separate compile because
           the rollout shape differs.
        4. Per-chunk-size `_batch_grad_fn` AND `pool.update_from_grad`
           scatter (snapshots/restores pool state).
        5. `pool.soft_update_targets` + `_compute_diagnostics`.
        6. Per-trainer meta-optim updates.

    Parameters
    ----------
    trainer : RuleTrainer
        The `RuleTrainer` to warm
    """

    def __init__(self, trainer: "RuleTrainer") -> None:
        self._t = trainer

    def run(self) -> Callable:
        """
        Execute all warm steps in order, ticking the dashboard 4 times.

        Returns
        -------
        meta_optim_update : Callable
            The (optionally JIT-wrapped) shared meta-optimizer update
            function for the trainer to store on
            `self._meta_optim_update`.
        """
        meta_params = self._t.meta_agent.get_params()
        meta_optim_update = self._meta_optim_update_fn()

        self._compiled_collect_phase(training=True)
        self._t.console.update_setup()

        self._compiled_collect_phase(training=False)
        self._t.console.update_setup()

        self._pool_grad_writeback(meta_params)
        self._t.console.update_setup()

        self._misc()
        self._t.console.update_setup()

        self._per_trainer_meta_optim(meta_params)
        jax.effects_barrier()

        return meta_optim_update

    def _meta_optim_update_fn(self) -> Callable:
        """
        Build (and optionally JIT) the shared meta-optimizer update fn.

        Returns
        -------
        update_fn : Callable
            `optax.GradientTransformation.update`, wrapped in `jax.jit`
            when `trainer.jit_compile` is `True`.
        """
        return (
            self._t.meta_optim.update
            if not self._t.jit_compile
            else jax.jit(self._t.meta_optim.update)
        )

    def _compiled_collect_phase(self, *, training: bool) -> None:
        """
        Compile one collection phase's scan kernel.

        Train and valid phases have different rollout shapes (`n_updates
        * seq_len` total steps with rollout slot length `seq_len` for
        train; `2 * seq_len` total steps as a single rollout for valid)
        so each phase compiles to its own XLA program. Builds a fresh
        `CollectCarry` from the pool's current state with a constant
        PRNG so `pool._rng` is not consumed, calls
        `pool._compiled_collect`, and discards the result — pool state
        is unchanged.

        Parameters
        ----------
        training : bool
            `True` to warm the training scan (uses `train_buffer`);
            `False` to warm the validation scan (uses `valid_buffer`).
        """
        pool = self._t.pool
        cfg = self._t.config

        if training:
            buffer = pool.train_buffer
            total_steps, seq_len = cfg.n_updates * cfg.seq_len, cfg.seq_len
        else:
            buffer = pool.valid_buffer
            valid_seq_len = cfg.seq_len * 2
            total_steps, seq_len = valid_seq_len, valid_seq_len

        init_carry = CollectCarry(
            obs=pool.obs,
            env_states=pool.env_states,
            prev_actions=pool.prev_actions,
            p_ocm_h=pool.p_ocm_h,
            p_acm_h=pool.p_acm_h,
            t_ocm_h=pool.t_ocm_h,
            t_acm_h=pool.t_acm_h,
            v_h=pool.v_h,
            rng=jax.random.key(0),
            write_idx=jnp.int32(0),
            buf=buffer.as_pytree(),
        )

        _, all_rewards, all_dones = pool._compiled_collect(
            self._t._batched_collect_fn,
            init_carry,
            total_steps,
            seq_len,
            pool.p_params,
            pool.t_params,
            pool.v_params,
            pool.action_masks,
        )
        _ = jax.device_get((all_rewards, all_dones))
        jax.effects_barrier()

    def _pool_grad_writeback(self, meta_params: chex.ArrayTree) -> None:
        """
        Warm `_batch_grad_fn` per chunk size AND `pool.update_from_grad`.

        Runs the actual gradient function and routes its output through
        `pool.update_from_grad` to compile the chunk-shaped scatter
        kernels. Snapshots pool state before and restores after so the
        warm-up doesn't bleed mutations into training.

        Parameters
        ----------
        meta_params : chex.ArrayTree
            Current meta-network parameters, shared across the chunk
            via `in_axes=None` in `_batch_grad_fn`'s vmap signature.
        """
        pool = self._t.pool
        chunk_sizes = {min(self._t.max_group_size, self._t.num_trainers)}
        remainder = self._t.num_trainers % self._t.max_group_size
        if remainder and remainder != min(self._t.max_group_size, self._t.num_trainers):
            chunk_sizes.add(remainder)

        snap = (
            pool.p_params,
            pool.v_params,
            pool.t_params,
            pool.v_opt,
            pool.adv_ema,
            pool.td_ema,
        )

        for chunk_size in chunk_sizes:
            disco_h, meta_h = [], []
            for idx in range(chunk_size):
                d_h, m_h = self._t.state.hidden.get(idx)
                disco_h.append(d_h)
                meta_h.append(m_h)
            grad_inputs = pool.get_grad_inputs(
                0,
                chunk_size,
                jnp.stack(disco_h),
                jnp.stack(meta_h),
            )
            chunk_out = self._t._batch_grad_fn(meta_params, *grad_inputs)
            pool.update_from_grad(0, chunk_size, chunk_out)
            del grad_inputs, chunk_out

        # Restore snapshots so warm-up doesn't bleed state into training
        (
            pool.p_params,
            pool.v_params,
            pool.t_params,
            pool.v_opt,
            pool.adv_ema,
            pool.td_ema,
        ) = snap
        jax.effects_barrier()

    def _misc(self) -> None:
        """
        Compile `pool.soft_update_targets` and `_compute_diagnostics`.

        Uses `tau=0.0` for the soft-target update so the actual state
        is unchanged but the kernel still traces. Diagnostics writes
        one harmless `meta_step=0` entry to the disco log group.
        """
        self._t.pool.soft_update_targets(0.0)
        self._t._compute_diagnostics()
        jax.effects_barrier()

    def _per_trainer_meta_optim(self, meta_params: chex.ArrayTree) -> None:
        """
        Warm the per-trainer meta-gradient optimizer updates.

        Each trainer holds its own Adan-with-clip optimizer state for
        meta-gradient normalization; each one compiles on first use.

        Parameters
        ----------
        meta_params : chex.ArrayTree
            Current meta-network parameters — used as the `params`
            argument to the optimizer's `update` fn.
        """
        dummy_grad = jax.tree.map(jnp.zeros_like, meta_params)
        for i in range(self._t.num_trainers):
            _ = self._t.pool.meta_optim_updates[i](
                dummy_grad,
                self._t.pool.meta_opt_states[i],
                meta_params,
            )
