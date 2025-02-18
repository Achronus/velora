![Logo](https://github.com/Achronus/velora/blob/main/assets/imgs/main.jpg)

[![codecov](https://codecov.io/gh/Achronus/velora/graph/badge.svg?token=OF7WP5Q9PT)](https://codecov.io/gh/Achronus/velora)
![Python Version](https://img.shields.io/pypi/pyversions/velora)
![License](https://img.shields.io/github/license/Achronus/velora)
![Issues](https://img.shields.io/github/issues/Achronus/velora)

Found on:

- [PyPi](https://pypi.org/project/velora)
- [GitHub](https://github.com/Achronus/velora)

# Velora

**Velora** is a lightweight and extensible framework built on top of powerful libraries like [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/), specializing in a unique approach to Deep Reinforcement Learning (RL) algorithms, a paradigm we call *Liquid RL*.

Instead of Fully-connected Networks, Velora combines [Liquid Neural Networks](https://arxiv.org/abs/2006.04439) (LNNs) with [Neural Circuit Policies](https://arxiv.org/abs/1803.08554) (NCPs), specifically [Ordinary Neural Circuits](https://proceedings.mlr.press/v119/hasani20a.html) (ONCs).

These two components have interesting benefits:

- LNNs are a powerful RNN architecture that learns system dynamics, not just data patterns.
- NCPs focus on sparsely connected neurons with distinct functions, mimicking biological behaviour.

From what we've seen, these networks are powerful, small-scale architectures that excel in model explainability, making them perfect for control tasks.

Velora offers Liquidfied PyTorch-based implementations of RL algorithms, designed to be intuitive, easy to use, and customizable.

In other frameworks, we've seen a trend of heavy abstraction in favour of minimal lines of code. Our approach aims to offer a best of both worlds, abstracting code away but making the details explainable on the backend, while giving you the freedom to customize as needed.

To get started, simply install it through [pip](https://pypi.org/):

```bash
pip install velora
```

Here's a simple example that should work 'as is':

```python
from functools import partial

from velora.models import LiquidDDPG
from velora.gym import wrap_gym_env
from velora.utils import set_device, set_seed

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, ClipReward

# Setup reproducibility and PyTorch device
seed = 64
set_seed(seed)

device = set_device()

# Add extra wrappers to our environment
env = wrap_gym_env("InvertedPendulum-v5", [
    partial(NormalizeObservation, epsilon=1e-8),
    partial(NormalizeReward, gamma=0.99, epsilon=1e-8),
    partial(ClipReward, max_reward=10.0),
    # RecordEpisodeStatistics,  # Applied automatically!
    # partial(NumpyToTorch, device=device),  # Applied automatically!
])

# Or, use the standard gym API (recommended for this env)
env = gym.make("InvertedPendulum-v5")

# Set core variables
state_dim = env.observation_space.shape[0]  # in features
n_neurons = 20  # decision/hidden nodes
action_dim = env.action_space.shape[0]  # out features

buffer_size = 100_000
batch_size = 128

# Train a model
model = LiquidDDPG(
    state_dim, 
    n_neurons, 
    action_dim, 
    buffer_size=buffer_size,
    device=device,
)
metrics = model.train(env, batch_size, n_episodes=300)
```

Currently, the framework only supports [Gymnasium](https://gymnasium.farama.org/) environments and will likely stay that way.

## API Structure

The frameworks API is designed to be simple and intuitive. We've broken into two main categories: `core` and `extras`.

### Core

The primary building blocks you'll use regularly.

```python
from velora.models import [algorithm]
```

### Extras

Utility methods that you may use occasionally.

```python
from velora.gym import [method]
from velora.utils import [method]
```

## Customization

Customization is at the heart of Velora but requires a deeper understanding of the API.

You can read more about it in the [documentation tutorials](https://velora.achronus.dev/learn/customize).

## Active Development

🚧 View the [Roadmap](https://velora.achronus.dev/starting/roadmap) 🚧

**Velora** is a tool that is continuously being developed. There's still a lot to do to make it a fully functioning framework, such as detailed API documentation, and more RL algorithms.

Our goal is to provide a quality open-source product that works 'out-of-the-box' that everyone can experiment with, and then gradually fix unexpected bugs and introduce more features on the road to a `v1` release.
