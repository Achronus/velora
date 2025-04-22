![Logo](https://raw.githubusercontent.com/Achronus/velora/main/docs/assets/imgs/main.jpg)

[![codecov](https://codecov.io/gh/Achronus/velora/graph/badge.svg?token=OF7WP5Q9PT)](https://codecov.io/gh/Achronus/velora)
![Python Version](https://img.shields.io/pypi/pyversions/velora)
![License](https://img.shields.io/github/license/Achronus/velora)
![Issues](https://img.shields.io/github/issues/Achronus/velora)

Found on:

- [PyPi](https://pypi.org/project/velora)
- [GitHub](https://github.com/Achronus/velora)

# Velora

**Velora** is a lightweight and modular framework built on top of powerful libraries like [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/). It is home to a new type of RL agent called ***NeuroFlow*** (NF) that specializes in Autonomous Cyber Defence through a novel Deep Reinforcement Learning (RL) approach we call ***Liquid RL***.

## Benefits

- **Explainability**: NF agents use [Liquid Neural Networks](https://arxiv.org/abs/2006.04439) (LNNs) and [Neural Circuit Policies](https://arxiv.org/abs/1803.08554) (NCPs) to model Cyber system dynamics, not just data patterns. Also, they use sparse NCP connections to mimic biological efficiency, enabling clear, interpretable strategies via a labeled Strategy Library.
- **Adaptability**: NF agents dynamically grow their networks using a fitness score, adding more neurons to a backbone only when new Cyber strategies emerge, keeping agents compact and robust.
- **Planning**: NF agents use a Strategy Library and learned environment model to plan strategic sequences for proactive Cyber defense.
- **Always Learning**: using [EWC](https://arxiv.org/abs/1612.00796), NF agents refine existing strategies and learn new ones post-training, adapting to evolving Cyber threats like new attack patterns.
- **Customizable**: NF agents are PyTorch-based, designed to be intuitive, easy to use, and modular so you can easily build your own!

## Installation

To get started, simply install it through [pip](https://pypi.org/project/velora) using one of the options below.

### GPU Enabled

For [PyTorch](https://pytorch.org/get-started/locally/) with CUDA (recommended):

```bash
pip install torch torchvision velora --extra-index-url https://download.pytorch.org/whl/cu126
```

### CPU Only

Or, for [PyTorch](https://pytorch.org/get-started/locally/) with CPU only:

```bash
pip install torch torchvision velora
```

## Example Usage

Here's a simple example that should work 'as is':

```python
from velora.models import NeuroFlow, NeuroFlowCT
from velora.utils import set_device

# Setup PyTorch device
device = set_device()

# For continuous tasks
model = NeuroFlowCT(
    "InvertedPendulum-v5",
    20,  # actor neurons 
    128,  # critic neurons
    device=device,
    seed=64,  # remove for automatic generation
)

# For discrete tasks
model = NeuroFlow(
    "CartPole-v1",
    20,  # actor neurons 
    128,  # critic neurons
    device=device,
)

# Train the model using a batch size of 64
model.train(64, n_episodes=50, display_count=10)
```

Currently, the framework only supports [Gymnasium](https://gymnasium.farama.org/) environments and is planned to expand to [PettingZoo](https://pettingzoo.farama.org/index.html) for Multi-agent (MARL) tasks, with updated adaptations of [CybORG](https://github.com/cage-challenge/CybORG/tree/main) environments.

## API Structure

The frameworks API is designed to be simple and intuitive. We've broken into two main categories: `core` and `extras`.

### Core

The primary building blocks you'll use regularly.

```python
from velora.models import [algorithm]
from velora.callbacks import [callback]
```

### Extras

Utility methods that you may use occasionally.

```python
from velora.gym import [method]
from velora.utils import [method]
```

## Active Development

🚧 View the [Roadmap](https://velora.achronus.dev/starting/roadmap) 🚧

**Velora** is a tool that is continuously being developed. There's still a lot to do to make it a great framework, such as detailed API documentation, and expanding our NeuroFlow agents.

Our goal is to provide a quality open-source product that works 'out-of-the-box' that everyone can experiment with, and then gradually fix unexpected bugs and introduce more features on the road to a `v1` release.
