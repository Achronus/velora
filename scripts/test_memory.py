"""
Memory leak regression test.

Runs N meta-steps and asserts that:
  1. GPU device memory does not grow linearly after warm-up.
  2. The XLA persistent cache does not accumulate new files during training
     (new files = new compilations = the concrete-closure fix isn't working).

Expected behaviour (after fix):
  - Step 0: first compilation — memory jumps to a working set.
  - Steps 1-2: possible minor warm-up growth.
  - Steps 3+: memory delta per step should be near zero.
  - Cache file count: fixed after _initial_setup; zero growth during training.

Run with:
    JAX_LOG_COMPILES=1 python scripts/test_memory.py

JAX_LOG_COMPILES=1 will print a line every time XLA recompiles something.
After the first couple of steps you should see *no* further compile messages.
"""

from pathlib import Path

import jax

from velora.disco import RuleTrainer, RuleTrainerSettings
from velora.disco.train import ParallelRuleTrainer
from velora.gym.envs import ATARI_BASE

# ── Configuration ──────────────────────────────────────────────────────────────
TEST_ENVS = ATARI_BASE
N_STEPS = 30
MAX_DELTA_MB = 50  # memory growth per step (after warm-up) must be below this
CACHE_DIR = Path(".cache/jax")


# ── Helpers ────────────────────────────────────────────────────────────────────
def bytes_in_use() -> int:
    """Return live bytes currently allocated on the default JAX device."""
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("bytes_in_use", 0)
    except Exception:
        # CPU-only builds don't expose memory_stats
        return 0


def cache_file_count() -> int:
    """Count files currently in the XLA persistent compilation cache."""
    if not CACHE_DIR.exists():
        return 0
    return sum(1 for _ in CACHE_DIR.rglob("*") if _.is_file())


# ── Instrumented training loop ─────────────────────────────────────────────────
samples: list[int] = []
cache_counts: list[int] = []

_original_apply = None


def _patch(trainer: RuleTrainer) -> None:
    """Monkey-patch _apply_meta_update to sample memory and cache after each meta-step."""
    from velora.disco.train import RuleTrainer as RT

    global _original_apply
    _original_apply = RT._apply_meta_update

    def hooked(self, avg_grad):
        _original_apply(self, avg_grad)  # type: ignore[operator]
        samples.append(bytes_in_use())
        cache_counts.append(cache_file_count())

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

    trainer = ParallelRuleTrainer(TEST_ENVS, config=config)
    _patch(trainer)

    try:
        trainer.train()
    finally:
        _unpatch()

    if not samples:
        print("No memory samples collected — is the training loop running?")
        return

    print(f"\nMemory samples ({len(samples)} steps):")
    for i, (s, c) in enumerate(zip(samples, cache_counts)):
        print(f"  step {i:3d}: {s / 1e6:.1f} MB  |  cache files: {c}")

    # ── Cache check ────────────────────────────────────────────────────────────
    # Step 0→1 may add a small number of files for operations not covered by
    # _initial_setup (e.g. meta_optim.update, apply_updates). That is expected.
    # From step 1 onwards the cache must be completely stable — any growth there
    # means new XLA programs are being compiled during training.
    if len(cache_counts) > 2:
        first_step_growth = cache_counts[1] - cache_counts[0]
        sustained_growth = cache_counts[-1] - cache_counts[1]
        print(f"\nXLA cache: +{first_step_growth} files on step 1 (expected first-use)")
        print(f"XLA cache: +{sustained_growth} files steps 1–{N_STEPS - 1} (must be 0)")

        assert sustained_growth == 0, (
            f"Recompilation detected: {sustained_growth} new XLA cache files "
            f"appeared after step 1 — the closure-constant fix may have regressed"
        )
        print("PASS: XLA cache is stable from step 1 onwards.")

    # ── Memory check ───────────────────────────────────────────────────────────
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
