# Getting Started

To get started, simply install it through [pip [:material-arrow-right-bottom:]](https://pypi.org/) using one of the options below.

For [PyTorch [:material-arrow-right-bottom:]](https://pytorch.org/get-started/locally/) with CUDA (recommended):

```bash
pip install torch torchvision velora --extra-index-url https://download.pytorch.org/whl/cu126
```

Or, for [PyTorch [:material-arrow-right-bottom:]](https://pytorch.org/get-started/locally/) with CPU only:

```bash
pip install torch torchvision velora
```

## Example Usage

Here's a simple example:

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

This code should work 'as is'.

Currently, the framework only supports [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) environments and is planned to expand to [PettingZoo [:material-arrow-right-bottom:]](https://pettingzoo.farama.org/index.html) for Multi-agent (MARL) tasks, with updated adaptations of [CybORG [:material-arrow-right-bottom:]](https://github.com/cage-challenge/CybORG/tree/main) environments.

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

-   :fontawesome-solid-droplet:{ .lg .middle } __NeuroFlow Agents__

    ---

    Learn how to use NeuroFlow models.

    [:octicons-arrow-right-24: Start learning](../learn/tutorial/index.md)

-   :material-puzzle-edit:{ .lg .middle } __Customize__

    ---

    Learn how to build your own models.

    [:octicons-arrow-right-24: Go custom](../learn/customize/index.md)

</div>
