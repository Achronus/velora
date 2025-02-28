# Saving & Loading Models

Saving a trained model and loading it are extremely common and useful practices when performing multiple experiments.

Both are really easy to do with our API and work identically for all agents.

Simply, select an agent you want to train, save it with it's instance `save` method and then `load` it the agent class method.

For our example, we'll use [`LiquidDDPG`](../tutorial/agents/ddpg.md).

## Saving a Model

To save a model we use the model instance's `save` method:

```python
from velora.models import LiquidDDPG
from velora.utils import set_seed, set_device

import gymnasium as gym

set_seed(64)
device = set_device()

env = gym.make("InvertedPendulum-v5")

model = LiquidDDPG(4, 10, 1, device=device)
metrics = model.train(env, 128, n_episodes=100)

model.save('checkpoints/ddpg_10n_100kb_100ep_gpu_invpendulum.pt')
```

This code should work 'as is'.

The only thing we need to do is give it a `filepath`. The complete model state will then be saved.

By default, we don't save the buffers state. However, if you want to, simply add `buffer=True` and it will store the buffer in a separate file.

```python
model.save('checkpoints/ddpg_10n_100kb_100ep_gpu_invpendulum.pt', buffer=True)
```

The buffer name will be identical to the filename, with a `.buffer` added between the filename and extension. So, with our previous example, the buffer file would save to:

> `checkpoints/ddpg_10n_100kb_100ep_gpu_invpendulum.buffer.pt`

## Loading a Model

To load a model we use the `load` class method:

```python
from velora.models import LiquidDDPG

model = LiquidDDPG.load('checkpoints/ddpg_10n_100kb_100ep_gpu_invpendulum.pt')
```

Like before, the only thing we need to do is give it a `filepath`. The complete model state will then be loaded into a new model instance.

Again, we don't load the buffers state by default. `buffer=True` will do that.

```python
model = LiquidDDPG.load(
    'checkpoints/ddpg_10n_100kb_100ep_gpu_invpendulum.pt',
    buffer=True,
)
```

???+ Warning "Buffer Loading"

    Buffer's can only be loaded if they have previously been saved with the same model state.

    The load method checks for the same `filepath` as the one provided but in buffer format: 
    
    > `<filepath>.buffer.<filepath_ext>`

    Theoretically, you could use a different `buffer` from what the has `model` been trained on by simply renaming a buffers filename to match the above format. 
    
    We personally haven't tested this but believe it's possible by how we've designed the `buffer` state (separate from `model` state). Feel free to experiment with this, it could create some interesting agent behaviour! 😉

## Filepath Naming Tips

Choosing the right name for a model `checkpoint` is always a difficult one. We encourage you to try different names that work best for you, but have a specific format that could be beneficial.

Here's our recommendation:

- All states should be stored in a `checkpoint` directory
- All states should have a clear definition of the parameters used

Core components to consider and include:

- `agent_name` - first item in the filepath (e.g., `ddpg`)
- `neuron_count` - followed by `n` (e.g., `10n`)
- `buffer_size` - followed by `b` (e.g., `100kb`)
- `episode_count` - followed by `ep` (e.g., `100ep`)
- `compute_device` - e.g., `gpu` or `cpu`
- `env_name` - environment name (e.g., `invpendulum`)

???+ abstract "Future Plans"

    We are looking into a better way to do this without the need for extremely long filenames.

    For `DDPG` we also need to consider its hyperparameters such as `gamma`, `tau`, `batch_size`, and `noise_scale`. Other models also have different hyperparameters. Where would it end?! 😭

    We'll likely convert this into a `config.yaml` file that is stored along with the `checkpoints`. It won't have any impact on the `save` and `load` methods but will be useful for quickly checking the model format.

---

Next, we'll start looking at each agent individually, starting with `DDPG`! 🚀
