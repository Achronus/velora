![Logo](https://raw.githubusercontent.com/Achronus/velora/main/docs/assets/imgs/main.jpg)

![Python Version](https://img.shields.io/pypi/pyversions/velora)
![License](https://img.shields.io/github/license/Achronus/velora)

# Velora

**Velora** is a Reinforcement Learning (RL) research framework for building autonomous systems that are lightweight, adaptive, explainable, persist in memory and can continuous learn.

It focuses on [Closed-form Continuous-time (CfC) Liquid Neural Networks](https://arxiv.org/abs/2106.13898) with **DiscoRL** (Discovery RL) — a meta-reinforcement learning algorithm that learns a *generalizable update rule* across a large suite of environments, rather than training a separate agent per task.

Traditional RL algorithms — such as PPO, DQN, or A3C — are hand-crafted by researchers and trained independently per environment. Each design decision (update rule, loss function, hyperparameters) requires careful manual tuning, making cross-task generalization difficult. DiscoRL sidesteps this by using meta-learning to *automatically discover* the update rule itself, producing one that is general-purpose by construction and outperforms manually designed rules across challenging benchmarks.

> **Note:** Our DiscoRL implementation currently only supports **discrete action spaces**. Continuous action space support is planned for a future release.

## Features

- **Meta-learning across many environments** — train over 57+ environments simultaneously with a shared update rule
- **JAX-native** — JIT compilation, `vmap` for parallel gradient computation, persistent XLA cache
- **Async CPU/GPU pipeline** — rollout collection for the next group runs on CPU while the GPU computes gradients for the current group
- **VRAM-efficient** — bfloat16 rollout buffers halve GPU memory usage; built-in VRAM advisor recommends the optimal batch size for your hardware
- **Parallel trainer** — `ParallelRuleTrainer` chunks environments into action-space groups and vmaps gradients across each chunk
- **Dashboard** — Rich-based live dashboard with progress, metrics, and losses; tqdm fallback for Docker/headless environments
- **Checkpointing** — orbax-based checkpoint saving and restoration

## Installation

Velora requires Python 3.13+ and [Flax](https://flax.readthedocs.io/en/latest/).

### GPU (recommended)

```bash
uv add velora jax[cuda13]
```

### CPU only

```bash
uv add velora
```

### TPU

```bash
uv add velora jax[tpu]
```

## Quick Start

### Sequential training (`RuleTrainer`)

Trains across environments one at a time. Good for debugging and smaller environment sets.

```python
from velora.disco import RuleTrainer, RuleTrainerSettings
from velora.gym.envs import ATARI_BASE, ATARI_EASY, EnvSet

envs = EnvSet(ATARI_BASE, ATARI_EASY)

config = RuleTrainerSettings(
    n_steps=1_000_000,
    n_updates=15,
    seq_len=29,
    num_vec_envs=4,
)

trainer = RuleTrainer(envs, config=config)
trainer.train()
```

### Parallel training (`ParallelRuleTrainer`)

Groups environments by action-space size and `vmaps` gradient computation across each group, overlapping CPU rollout collection with GPU gradient passes.

```python
from velora.disco import ParallelRuleTrainer, RuleTrainerSettings
from velora.gym.envs import ATARI_57, EnvSet

envs = EnvSet(ATARI_57)
config = RuleTrainerSettings(n_steps=1_000_000)

trainer = ParallelRuleTrainer(envs, config=config, max_group_size=8)
trainer.train()
```

## API Overview

```python
# Trainers
from velora.disco import RuleTrainer, ParallelRuleTrainer

# Configuration
from velora.disco import RuleTrainerSettings

# Environment sets (Atari)
from velora.gym.envs import ATARI_BASE, ATARI_EASY, ATARI_MEDIUM, ATARI_HARD, ATARI_57, EnvSet
```

### `RuleTrainerSettings` — key parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `n_steps` | `1_000_000` | Number of meta-training steps |
| `n_updates` | `15` | Inner-loop gradient updates per environment |
| `seq_len` | `29` | Rollout trajectory length |
| `num_vec_envs` | `4` | Parallel environment instances per trainer |
| `meta_lr` | `0.001` | Meta-optimizer learning rate |

### `ParallelRuleTrainer` — additional parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `max_group_size` | `8` | Max trainers per vmapped gradient chunk |
| `use_bfloat16` | `True` | Store rollout floats in `bfloat16` (halves VRAM) |
| `verbose` | `True` | Use Rich dashboard (`False` → `tqdm` alternative) |

## IDE Setup

For full IntelliSense support (auto-imports for all subpackage symbols), add the following to your project's `.vscode/settings.json`:

```json
{
    "python.analysis.packageIndexDepths": [
        {
            "name": "velora",
            "depth": 3,
            "includeAllSymbols": true
        }
    ]
}
```

## References

- Oh, J., Farquhar, G., Kemaev, I., Calian, D. A., Hessel, M., Zintgraf, L., Singh, S., van Hasselt, H., & Silver, D. (2025). Discovering state-of-the-art reinforcement learning algorithms. *Nature*, 648, 312–319. [https://doi.org/10.1038/s41586-025-09761-x](https://doi.org/10.1038/s41586-025-09761-x)

## Active Development

🚧 View the [Roadmap](https://velora.achronus.dev/starting/roadmap) 🚧

**Velora** is under active development. The DiscoRL algorithm, environment coverage, and documentation are all expanding.
