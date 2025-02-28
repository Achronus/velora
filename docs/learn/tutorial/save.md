# Saving & Loading Models

Saving a trained model and loading it are extremely common and useful practices when performing multiple experiments.

Both are really easy to do with our API and work identically for all agents.

Simply, select an agent you want to train, save it with it's instance `save` method and then `load` it with the agent class method.

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

model.save('checkpoints/ddpg/InvertedPendulum_final.pt')
```

This code should work 'as is'.

The only thing we need to do is give it a `filepath`. The complete model state will then be saved along with a `model_config.json` file that provides comprehensive details about the trained agent.

By default, we don't save the buffers state. However, if you want to, simply add `buffer=True` and it will store the buffer in a separate file.

```python
model.save('checkpoints/ddpg/InvertedPendulum_final.pt', buffer=True)
```

The buffer name will be identical to the filename, with a `.buffer` added between the filename and extension. So, with our previous example, the buffer file would save to:

> `checkpoints/ddpg/InvertedPendulum_final.buffer.pt`

Similarly, the `model_config.json` would save to:

> `checkpoints/ddpg/model_config.json`

## Loading a Model

To load a model we use the `load` class method:

```python
from velora.models import LiquidDDPG

model = LiquidDDPG.load('checkpoints/ddpg/InvertedPendulum_final.pt')
```

Like before, the only thing we need to do is give it a `filepath`. The complete model state will then be loaded into a new model instance.

???+ note "model_config.json"

    The `model_config.json` is not loaded into the model. 
    
    This file is only used for quickly checking the details of a trained agent when looking through the `checkpoint` folder.

Again, we don't load the buffers state by default. `buffer=True` will do that.

```python
model = LiquidDDPG.load(
    'checkpoints/ddpg/InvertedPendulum_final.pt',
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

- All states should be stored in a `checkpoint` directory.
- All states should have a clear `folder` for their algorithm.
- All states should start with their `gymnasium.Env` name and either end in `ep[count]` or `final`.

---

Next, we'll start looking at each agent individually, starting with `DDPG`! 🚀
