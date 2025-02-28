# Using Callbacks

???+ api "API Docs"

    [`velora.callbacks`](../reference/callbacks.md)

The normal process for training an agent is extremely limited. There is no ability to stop at a reward threshold or save a model's state.

Let's be honest, do you really want to sit through 100k episodes and then manually have to save your model, even though it solved the environment at say 10k episodes? I know I don't! 😅

Callbacks are a flexible and powerful way to change that.

When calling the `train()` method you can use the `callbacks` parameter to extend the functionality of the training process.

## Basic Usage

For example, let's say we want to stop our agent when it achieves an average reward of `15`.

To do this, we use the `EarlyStopping` callback:

```python
from velora.callbacks import EarlyStopping
from velora.models import LiquidDDPG

import gymnasium as gym


env = gym.make("InvertedPendulum-v5")
model = LiquidDDPG(4, 10, 1)

callbacks = [
    EarlyStopping(target=15.0, patience=3),
]

metrics = model.train(env, 128, n_episodes=100_000, callbacks=callbacks)
```

This code should work 'as is'.

Now, the model will automatically terminate when it reaches the reward target! It's as simple as that! 😊

## Callback List

??? tip "Combining Callbacks"

    Callbacks alone are extremely powerful for enhancing your training process, but it becomes even more ridiculous when you stack them together! E.g.,

    ```python
    from velora.callbacks import EarlyStopping, SaveCheckpoints

    # ...

    callbacks = [
        EarlyStopping(15.0),
        SaveCheckpoints(model, "agent"),
    ]

    metrics = model.train(..., callbacks=callbacks)
    ```

    We highly recommend you experiment with different `callbacks` yourself and find what ones work best for you.
    
    The possibilities are truly endless! 😎

There are a number of `callbacks` available. Here's an exhaustive list of them:

- [`EarlyStopping`](#early-stopping) - stops the training process when the average reward `target` is reached multiple times in a row based on the `patience` value.
- [`SaveCheckpoints`](#model-checkpoints) - saves the model state during the training process based on a `frequency`.

## Early Stopping

???+ api "API Docs"

    [`velora.callbacks.EarlyStopping(target, patience)`](../reference/callbacks.md#velora.callbacks.EarlyStopping)

`EarlyStopping` terminates the training process when the average reward `target` is reached multiple times in a row based on the `patience` value.

```python
from velora.callbacks import EarlyStopping

callbacks = [
    EarlyStopping(target=15.0, patience=3),
]
```

By default, the `patience=3` and is the only optional parameter available.

## Model Checkpoints

???+ api "API Docs"

    [`velora.callbacks.SaveCheckpoints(agent, dirname)`](../reference/callbacks.md#velora.callbacks.SaveCheckpoints)

Sometimes it can be really useful to save the model state intermittently during the training process, especially when you are also using `EarlyStopping`.

We can do this with the `SaveCheckpoints` callback. It requires two parameters:

- `agent` - the model used during training.
- `dirname` - the directory name to store the model checkpoints in the `checkpoints` directory.

For example, if we want to train a `DDPG` model and store its checkpoints in `checkpoints/ddpg` we'd use the following code:

```python
from velora.callbacks import SaveCheckpoints
from velora.models import LiquidDDPG

model = LiquidDDPG(4, 10, 1)

callbacks = [
    SaveCheckpoints(model, "ddpg"),
]
```

Notice how we don't allow you to set a `prefixed` name for checkpoints. It's set automatically with the environment name and episode count, such as:

- `InvertedPendulum_ep100.pt`
- `InvertedPendulum_final.pt`

We limit your control to the directory name to simplify the checkpoint process and to keep them organised.

??? question "Why only the `dirname`?"

    Under the hood, we use the `model.save()` method for storing checkpoints, so each checkpoint folder will also contain a `model_config.json` file containing comprehensive details of the trained agent. 
    
    That way, you don't need any complex `dirnames`! 😉

Only setting the required parameters will create an instance of the callback with the following default `optional` parameters:

- `frequency=100` - the episode save frequency.
- `buffer=False` - whether to save the final buffer state.

You can customize these freely using the required parameter name.

When `buffer=True` only the final checkpoint state's `buffer` is saved. We'll discuss more about this in the [Saving and Loading Models](../tutorial/save.md) section.

## Analytics

!!! construction "Coming Soon"

## Recording Episodes

!!! construction "Coming Soon"

---

Next, we'll look at how to `save` and `load` models. See you there! 👋
