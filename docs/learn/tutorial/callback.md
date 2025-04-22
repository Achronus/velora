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
from velora.models import NeuroFlow

model = NeuroFlow("InvertedPendulum-v5", 20, 128)

callbacks = [
    EarlyStopping(target=15.0, patience=3),
]

model.train(128, callbacks=callbacks)
```

This code should work 'as is'.

Now, the model will automatically terminate when it reaches the reward target! It's as simple as that! 😊

### Callback List

??? tip "Combining Callbacks"

    Callbacks alone are extremely powerful for enhancing your training process, but it becomes even more ridiculous when you stack them together! E.g.,

    ```python
    from velora.callbacks import EarlyStopping, SaveCheckpoints

    # ...

    callbacks = [
        EarlyStopping(15.0),
        SaveCheckpoints("agent"),
    ]

    model.train(..., callbacks=callbacks)
    ```

    We highly recommend you experiment with different `callbacks` yourself and find what ones work best for you.
    
    The possibilities are truly endless! 😎

There are a number of `callbacks` available. Here's an exhaustive list of them:

- [`EarlyStopping`](#early-stopping) - stops the training process when the average reward `target` is reached multiple times in a row based on the `patience` value.
- [`SaveCheckpoints`](#model-checkpoints) - saves the model state during the training process based on a `frequency`.
- [`RecordVideos`](#recording-videos) - adds video recording to the agents training process.
- [`CometAnalytics`](#comet-analytics) - adds [Comet [:material-arrow-right-bottom:]](https://www.comet.com/) experiment cloud-based tracking.

## Early Stopping

???+ api "API Docs"

    [`velora.callbacks.EarlyStopping(target, patience)`](../reference/callbacks.md#velora.callbacks.EarlyStopping)

`EarlyStopping` terminates the training process when the episodic reward `target` is reached multiple times in a row based on the `patience` value.

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

For example, if we want to train a `NeuroFlow` model and store its checkpoints in a model directory called `nf` we'd use the following code:

```python
from velora.callbacks import SaveCheckpoints

callbacks = [
    SaveCheckpoints("nf"),
]
```

Notice how we don't allow you to set a `prefixed` name for checkpoints. It's set automatically with the environment name and episode count, such as:

- `InvertedPendulum_100/`
- `InvertedPendulum_final/`

We limit your control to the directory name to simplify the checkpoint process and to keep them organised.

??? question "Why only the `dirname`?"

    Under the hood, we use the `agent.save()` method for storing checkpoints (more on this later). It stores a variety of state files and two additional ones in the `saves` folder - `model_config.json` a file containing config details about the agent, and a `completed.json` file after training terminates with final stats and duration (`episodes`, `steps` and `time taken`).
    
    That way, you don't need any complex `dirnames`! 😉

Only setting the required parameters will create an instance of the callback with the following default `optional` parameters:

- `frequency=100` - the `episode` save frequency.
- `buffer=False` - whether to save the buffer state.

You can customize these freely using the required parameter name.

When `buffer=True` the checkpoint state's `buffer` is saved at that episode. We'll discuss more about this in the [Saving and Loading Models](../tutorial/save.md) section.

## Recording Videos

???+ api "API Docs"

    [`velora.callbacks.RecordVideos(dirname)`](../reference/callbacks.md#velora.callbacks.RecordVideos)

Sometimes it's useful to see how the agent is performing while it is training. The best way to do this is visually, by watching the agent interact with its environment.

To do this, we use the `RecordVideos` callback. Under-the-hood, we apply [Gymnasium's RecordVideo [:material-arrow-right-bottom:]](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo) wrapper to the environment with a minor expansion - it *always* records the final training episode.

It has one required parameter:

- `dirname` - the model directory name to store the videos. E.g., `nf`.

Videos are automatically added to a `checkpoints` directory inside a `<dirname>/videos` folder. This design choice compliments the [`SaveCheckpoints`](#model-checkpoints) callback to help keep the experiments tidy.

```python
from velora.callbacks import EarlyStopping, SaveCheckpoints, RecordVideos

# Solo
callbacks = [
    RecordVideos("nf"),
]

# With other callbacks
CP_DIR = "nf"
FREQ = 5

callbacks = [
    SaveCheckpoints(CP_DIR, frequency=FREQ, buffer=True),
    EarlyStopping(target=15., patience=10),
    RecordVideos(CP_DIR, frequency=FREQ),
]
```

This code should work 'as is'.

For more control, you can also set any of the optional parameters:

- `method=episode` - the recording method. Either: `episode` or `step`.
- `frequency=100` - the record frequency for `method`.

## Analytics

Experiment tracking is extremely important for understanding how an agent is performing. We offer two variants of this: `offline` and `online` (cloud-based).

Our offline approach works out of the box with every Velora agent. We'll talk about this more in the [Training Metrics section](../tutorial/metrics.md).

However, online (cloud-based) tracking is optional. To integrate it we use callbacks! 😊

### Comet Analytics

???+ api "API Docs"

    [`velora.callbacks.CometAnalytics(project_name)`](../reference/callbacks.md#velora.callbacks.CometAnalytics)

We've found [Comet [:material-arrow-right-bottom:]](https://www.comet.com/) to be one of the best tools for RL experiments. It has a clean interface, an elegant category system for experiment details, and integrates well with video recordings.

To use it, we need 3 things -

1. The required dependencies:

    ```bash title=""
    pip install velora[comet]
    ```

2. A `COMET_API_KEY` (found in your Comet account settings [API Key Docs [:material-arrow-right-bottom:]](https://www.comet.com/docs/v2/api-and-sdk/rest-api/overview/#obtaining-your-api-key)):

    You can either configure this using an `.env` file or setting it manually in the terminal -

    === "`.env` file"

        ```bash title=""
        COMET_API_KEY=
        ```

    === "Linux/macOS"

        ```bash title=""
        export COMET_API_KEY=
        ```

    === "Windows (CMD)"

        ```bash title=""
        set COMET_API_KEY=
        ```

    === "Powershell"

        ```powershell title=""
        $env:COMET_API_KEY="<insert_here>"
        ```

3. The dedicated callback - `CometAnalytics`:

    ```python
    from velora.callbacks import CometAnalytics

    callbacks = [
        # other callbacks
        CometAnalytics("nf"),
    ]
    ```

The callback has one required parameter:

- `project_name` - the name of the Comet ML project to add the experiment to.

And three optional parameters:

- `experiment_name` - the name of the experiment. If `None`, automatically creates the name using the format: `<agent_classname>_<env_name>_<n_episodes>ep`.

    > E.g., `NeuroFlow_InvertedPendulum_1000ep`.

- `tags` - a list of tags associated with experiment. If `None`, sets tags automatically as: `[agent_classname, env_name]`.

???+ note

    We've limited the customization to keep things simple. By default, the experiment will be tied to your account associated to the `COMET_API_KEY`.

    You shouldn't have to tweak a million settings just to start tracking experiments! 🚀

We primarily focus on sending episodic metrics to Comet that provide a detailed overview of the training process. These include:

- `episode/return` - the raw episodic reward (return).
- `episode/length` - the number of steps completed in the episode.
- `reward/moving_avg` - the episodic reward moving average based on the training `window_size`.
- `reward/moving_upper` - the episodic reward moving upper bound (`moving_avg + moving_std`) based on the training `window_size`.
- `reward/moving_lower` - the episodic reward moving lower bound (`moving_avg - moving_std`) based on the training `window_size`.
- `losses/actor_loss` - the average Actor loss for each episode.
- `losses/critic_loss` - the average Critic loss for each episode.
- `losses/entropy_loss` - the average Entropy loss for each episode.

---

Next, we'll look at how to `save` and `load` models. See you there! 👋
