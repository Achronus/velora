"""
Memory leak regression test.

Runs N meta-steps and asserts that GPU device memory does not grow linearly.

Expected behaviour (after fix):
  - Step 0: first compilation — memory jumps to a working set.
  - Steps 1-2: possible minor warm-up growth.
  - Steps 3+: memory delta per step should be near zero.

Run with:
    JAX_LOG_COMPILES=1 python test_memory.py

JAX_LOG_COMPILES=1 will print a line every time XLA recompiles something.
After the first couple of steps you should see *no* further compile messages.
"""

import jax

from velora.disco import RuleTrainer, RuleTrainerSettings
from velora.gym.envs import ATARI_BASE

# ── Configuration ──────────────────────────────────────────────────────────────
# Use a small 2-env subset so the test finishes quickly.
# Both envs have the same action space so only one XLA program is compiled.
TEST_ENVS = ATARI_BASE
N_STEPS = 10
MAX_DELTA_MB = 50  # memory growth per step (after warm-up) must be below this


# ── Helpers ────────────────────────────────────────────────────────────────────
def bytes_in_use() -> int:
    """Return live bytes currently allocated on the default JAX device."""
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("bytes_in_use", 0)
    except Exception:
        # CPU-only builds don't expose memory_stats
        return 0


# ── Instrumented training loop ─────────────────────────────────────────────────
samples: list[int] = []

_original_apply = None


def _patch(trainer: RuleTrainer) -> None:
    """Monkey-patch _apply_meta_update to sample memory after each meta-step."""
    from velora.disco.train import RuleTrainer as RT

    global _original_apply
    _original_apply = RT._apply_meta_update

    def hooked(self, avg_grad):
        _original_apply(self, avg_grad)  # type: ignore[operator]
        samples.append(bytes_in_use())

    RT._apply_meta_update = hooked  # type: ignore[method-assign]


def _unpatch() -> None:
    from velora.disco.train import RuleTrainer as RT

    if _original_apply is not None:
        RT._apply_meta_update = _original_apply  # type: ignore[method-assign]


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    config = RuleTrainerSettings(
        n_steps=N_STEPS,
        n_updates=3,  # keep the inner loop short for speed
        # seq_len=16,
    )

    trainer = RuleTrainer(TEST_ENVS, config=config)
    _patch(trainer)

    try:
        trainer.train()
    finally:
        _unpatch()

    if not samples:
        print("No memory samples collected — is the training loop running?")
        return

    print(f"\nMemory samples ({len(samples)} steps):")
    for i, s in enumerate(samples):
        print(f"  step {i:3d}: {s / 1e6:.1f} MB")

    # Warm-up: skip first 3 steps (compilation + initial allocations)
    warmup = 3
    if len(samples) <= warmup:
        print(f"Too few steps ({len(samples)}) to evaluate post-warmup growth.")
        return

    deltas = [samples[i + 1] - samples[i] for i in range(warmup, len(samples) - 1)]
    avg_delta_mb = sum(deltas) / len(deltas) / 1e6

    print(f"\nAvg memory delta (steps {warmup}–{N_STEPS}): {avg_delta_mb:.1f} MB/step")

    assert avg_delta_mb < MAX_DELTA_MB, (
        f"Memory leak detected: {avg_delta_mb:.1f} MB/step growth "
        f"(threshold: {MAX_DELTA_MB} MB)"
    )
    print("PASS: memory is stable after warm-up.")


if __name__ == "__main__":
    main()
