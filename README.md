![Logo](https://raw.githubusercontent.com/Achronus/velora/main/docs/assets/imgs/main.jpg)

![Python Version](https://img.shields.io/pypi/pyversions/velora)
![License](https://img.shields.io/github/license/Achronus/velora)

Found on:

- [PyPi](https://pypi.org/project/velora)
- [GitHub](https://github.com/Achronus/velora)

> 🚧 Velora is under active development and its API, documentation and examples may change overtime. 🚧

# Velora

**Velora** is a Reinforcement Learning (RL) research framework for exploring ways to build lightweight, adaptable, transparent and stateful agents that move away from the world of Large Language Models (LLMs).

By design, it focuses on tasks centred around robotics and continuous control problems to bring us closer to unlocking physical agents that are useful for real-world use cases.

Built with PyTorch, it provides modular building blocks that plug into common RL agent algorithms (such as [PPO](https://arxiv.org/abs/1707.06347) and [TD3](https://arxiv.org/abs/1802.09477)) and environment backends like [MJWarp](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) for rapid iteration and experimentation.

Velora is **not** a replacement for other popular RL libraries (such as [Stable Baselines3](https://sb3-contrib.readthedocs.io/en/master/index.html) or [RLlib](https://docs.ray.io/en/latest/rllib/index.html)) and is purely a framework for experimenting with unconventional models and techniques that have real potential.

For more details on how it works and what's inside the framework, refer to our [documentation](https://velora.achronus.dev/).

## Package versions

| Package     | Version                                   |
| ----------- | ----------------------------------------- |
| Python      | `3.12`                                    |
| PyTorch     | `2.11.0` (CUDA 13)                        |
| TorchVision | `0.26.0`                                  |
| Isaac Lab   | `3.0.0b2.post1` (optional, `isaac` extra) |

## Installation

Velora only supports Python 3.12.

> [!NOTE]
> The index configuration below only applies when installing Velora from PyPI. If you clone the repository and use `uv sync`, the CUDA 13 wheels are installed automatically.

### uv

Configure the PyTorch index in your project's `pyproject.toml` first, so `torch` resolves to the CUDA 13 wheels:

```toml
[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cuda" }
torchvision = { index = "pytorch-cuda" }
```

Then add the package:

```bash
uv add velora
```

> [!TIP]
> Already ran `uv add velora`? Add the configuration above, then run `uv sync` to re-resolve against the new index.

### pip

```bash
pip install velora
```

This installs PyTorch from PyPI's standard wheels. For the CUDA 13 builds, add the PyTorch `cu130` index:

```bash
pip install velora --extra-index-url https://download.pytorch.org/whl/cu130
```

## Simulation (Isaac Lab)

Isaac Lab and Isaac Sim are optional and require a NVIDIA GPU. Install them with the `isaac` extra.

### uv

Add the NVIDIA index to your project's `pyproject.toml` alongside the PyTorch one:

```toml
[[tool.uv.index]]
name = "nvidia"
url = "https://pypi.nvidia.com"
explicit = true

[tool.uv.sources]
isaacsim = { index = "nvidia" }
isaaclab = { index = "nvidia" }
```

Then add the extra:

```bash
uv add "velora[isaac]"
```

### pip

```bash
pip install "velora[isaac]" --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com
```
