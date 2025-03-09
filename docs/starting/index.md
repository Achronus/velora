# Getting Started

To get started, simply install it through [pip [:material-arrow-right-bottom:]](https://pypi.org/) using one of the options below.

For [PyTorch [:material-arrow-right-bottom:]](https://pytorch.org/get-started/locally/) with CUDA (recommended):

```bash
pip install torch torchvision velora --extra-index-url https://download.pytorch.org/whl/cu124
```

Or, for [PyTorch [:material-arrow-right-bottom:]](https://pytorch.org/get-started/locally/) with CPU only:

```bash
pip install velora
```

## Example Usage

Here's a simple example:

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

This code should work 'as is'.

Currently, the framework only supports [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) environments and is planned to expand to [PettingZoo [:material-arrow-right-bottom:]](https://pettingzoo.farama.org/index.html) for Multi-agent (MARL) tasks.

## API Structure

The frameworks API is designed to be simple and intuitive. We've broken into two main categories: [`core`](#core) and [`extras`](#extras).

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

## Next Steps

<div class="grid cards" markdown>

-   :fontawesome-solid-droplet:{ .lg .middle } __Use a Liquid RL Model__

    ---

    Learn how to use Liquid RL models with Gymnasium environments.

    [:octicons-arrow-right-24: Start learning](../learn/tutorial/index.md)

-   :material-puzzle-edit:{ .lg .middle } __Customize__

    ---

    Learn how to build your own custom models.

    [:octicons-arrow-right-24: Start learning](../learn/customize/index.md)

</div>
