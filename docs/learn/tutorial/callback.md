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
- [`RecordVideos`](#recording-videos) - adds video recording to the agents training process.

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

    [`velora.callbacks.SaveCheckpoints(dirname)`](../reference/callbacks.md#velora.callbacks.SaveCheckpoints)

Sometimes it can be really useful to save the model state intermittently during the training process, especially when you are also using `EarlyStopping`.

We can do this with the `SaveCheckpoints` callback. It requires one parameter:

- `dirname` - the directory name to store the model checkpoints in the `checkpoints` directory.

Checkpoints are automatically added to a `checkpoints` directory inside a `<dirname>/saves` folder. This design choice compliments the [`RecordVideos`](#recording-videos) callback to help keep the experiments tidy.

For example, if we want to train a `DDPG` model and store its checkpoints in a model directory called `ddpg` we'd use the following code:

```python
from velora.callbacks import SaveCheckpoints

callbacks = [
    SaveCheckpoints("ddpg"),
]
```

Notice how we don't allow you to set a `prefixed` name for checkpoints. It's set automatically with the environment name and episode count, such as:

- `InvertedPendulum_ep100.pt`
- `InvertedPendulum_final.pt`

We limit your control to the directory name to simplify the checkpoint process and to keep them organised.

??? question "Why only the `dirname`?"

    Under the hood, we use the `agent.save()` method for storing checkpoints (more on this later), so each checkpoint folder will also contain a `model_config.json` file containing comprehensive details of the trained agent.
    
    That way, you don't need any complex `dirnames`! 😉

Only setting the required parameters will create an instance of the callback with the following default `optional` parameters:

- `frequency=100` - the episode save frequency.
- `buffer=False` - whether to save the final buffer state.

You can customize these freely using the required parameter name.

When `buffer=True` only the final checkpoint state's `buffer` is saved. We'll discuss more about this in the [Saving and Loading Models](../tutorial/save.md) section.

## Recording Videos

???+ api "API Docs"

    [`velora.callbacks.RecordVideos(method, dirname)`](../reference/callbacks.md#velora.callbacks.RecordVideos)

Sometimes it's useful to see how the agent is performing while it is training. The best way to do this is visually, by watching the agent interact with its environment.

Normally, you would use [Gymnasium's RecordVideo [:material-arrow-right-bottom:]](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo) wrapper for this, but instead, we recommend you use the `RecordVideos` callback.

It uses the same approach as the wrapper but integrates seamlessly with other callbacks and adds a minor expansion - it *always* records the final training episode.

It takes in two parameters:

- `method` - the recording method. Either: `episodes` or `steps`.
- `dirname` - the model directory name to store the videos. E.g., `ddpg`.

Videos are automatically added to a `checkpoints` directory inside a `<dirname>/videos` folder. This design choice compliments the [`SaveCheckpoints`](#model-checkpoints) callback to help keep the experiments tidy.

```python
from velora.callbacks import EarlyStopping, SaveCheckpoints, RecordVideos

# Solo
callbacks = [
    RecordVideos("episode", "ddpg"),
]

# With other callbacks
CP_DIR = "ddpg"
FREQ = 5

callbacks = [
    SaveCheckpoints(CP_DIR, frequency=FREQ, buffer=True),
    EarlyStopping(target=15.),
    RecordVideos("episode", CP_DIR, frequency=FREQ),
]
```

This code should work 'as is'.

`frequency` is an optional parameter. If not set, it will default to `100`.

## Analytics

!!! construction "Coming Soon"
<!-- 
???+ api "API Docs"

    [`velora.callbacks.SaveCheckpoints(agent, dirname)`](../reference/callbacks.md#velora.callbacks.SaveCheckpoints)

```bash
pip install velora[analytics]
``` -->

---

Next, we'll look at how to `save` and `load` models. See you there! 👋
