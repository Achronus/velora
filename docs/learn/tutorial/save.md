# Saving & Loading Models

Saving a trained model and loading it are extremely common and useful practices when performing multiple experiments.

Both are really easy to do with our API and work identically for all agents.

Simply, select an agent you want to train, save it with it's instance `save` method and then `load` it with the agent class method.

For our example, we'll use [`NeuroFlow`](../tutorial/agents/nf.md).

## Saving a Model

???+ api "API Docs"

    [`velora.models.base.RLModuleAgent.save(dirpath)`](../reference/models/base.md#velora.models.base.RLModuleAgent.save)

To save a model we use the model instance's `save` method:

```python
from velora.models import NeuroFlow
from velora.utils import set_device

device = set_device()

model = NeuroFlow("InvertedPendulum-v5", 20, 128, device=device, seed=64)
model.train(128, n_episodes=10, window_size=5)

model.save('checkpoints/nf/saves/InvertedPendulum_10')
```

This code should work 'as is'.

The only thing we need to do is give it a `dirpath` where the last folder contains all the models state files. These include:

- `metadata.json` - contains the model and optimizer metadata.
- `model_state.safetensors` - contains the model weights and biases.
- `optim_state.safetensors` - contains the optimizer states (actor and critic).

Optionally, we can also save the buffer state with `buffer=True`:

- `buffer_state.safetensors` - contains the buffer state.
- `metadata.json` - extended to include the `buffer` metadata.

```python
model.save(..., buffer=True)
```

And/or, optionally, the model config with `config=True` (stored in the `dirpath.parent`):

- `model_config.json` - contains the core details of the agent.

```python
model.save(..., config=True)
```

??? question "Why the parent directory?"

    The `model_config.json` contains comprehensive details about the agent (its `model.config`). It never changes. Therefore, it should only be saved once.

    Typically, you'll save a model state during the training process after `n_episodes` (just like we do with the [SaveCheckpoints](../tutorial/callback.md#model-checkpoints) callback).

    The file is only used to provide an overview of the model so you can easily identify an experiment without having to manually load a model's state. So, we store it above the `target` directory with the assumption that you are saving more than once for the same experiment.

Notice how we are using [safetensors [:material-arrow-right-bottom:]](https://github.com/huggingface/safetensors). This helps us maximize tensor security and performance! 😉

## Loading a Model

???+ api "API Docs"

    [`velora.models.base.RLModuleAgent.load(dirpath)`](../reference/models/base.md#velora.models.base.RLModuleAgent.load)

To load a model we use the `load` class method:

```python
from velora.models import NeuroFlow

model = NeuroFlow.load('checkpoints/nf/saves/InvertedPendulum_10')
```

Like before, the only thing we need to do is give it a `dirpath`. The complete model state will then be loaded into a new model instance.

Again, we don't load the buffers state by default. `buffer=True` will do that.

```python
model = NeuroFlow.load(..., buffer=True)
```

???+ Warning "Buffer Loading"

    Buffer's can only be loaded if they have previously been saved with the same model state.

    The load method checks for a `buffer_state.safetensors` and a `metadata.json` file with the `buffer` metadata.

---

Next, we'll start looking at each agent individually! 🚀
