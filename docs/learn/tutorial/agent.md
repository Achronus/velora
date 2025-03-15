# Agent Basics

RL agents are at the heart of Velora's framework and they are one of the main features that you'll often use for your experiments.

Each agent has suitable differences but we've designed them to act like drop-in replacements of each other. Their underlying functionality may differ, but their core API is identical with the exception of optional hyperparameters.

At it's core, you have three main operations:

- **Initialize** - creation of the model.
- **Training** - running the model on an environment.
- **Prediction** - making predictions on unseen data.

## Creating Agents

Creating a agent is really easy! We simply declare our agent from the `velora.models` API and create a class instance.

Each model requires three main parameters:

1. `state_dim` - the number of input nodes
2. `n_neurons` - the number of hidden nodes
3. `action_dim` - the number of output nodes

And that's it! Here's an example:

```python
from velora.models import LiquidDDPG

model = LiquidDDPG(4, 10, 1)
```

This code should work 'as is'.

Want to use a different agent? Just swap out `LiquidDDPG` with a different one!

???+ warning "LiquidPPO"

    `LiquidPPO` will be introduced in a later version of the framework. 
    
    We use it here strictly for demonstration purposes.

```python
from velora.models import LiquidPPO

model = LiquidPPO(4, 10, 1)
```

It really is that easy! 🤩

???+ note "Agent Parameters"

    Each agent comes with a set of optional parameters that can be customized. 
    
    You can read more about them in the [`Agents`](../tutorial/agents/index.md) documentation section.

## Training an Agent

Training an agent is just as easy!

We use the `train` method, supply it with a `Gymnasium` environment and a `batch_size` and boom 💥, your agent will start training for `1000` episodes:

```python
from velora.models import LiquidDDPG
import gymnasium as gym

env = gym.make('InvertedPendulum-v5')

model = LiquidDDPG(4, 10, 1)
model.train(env, 128)
```

This code should work 'as is'.

Want to change the number of episodes? Use the `n_episodes` parameter! What about the episodic training status rate? Use the `window_size` parameter!

```python
model.train(env, 128, n_episodes=10_000, window_size=1000)
```

These are two core optional parameters for the `train()` method, with the addition of [`callbacks`](../tutorial/callback.md) but we'll talk about them later.

Each agent has it's own set of unique hyperparameters for it's `train()` method. They vary depending on the algorithm, so we'll leave this for a later section! 😉

??? tip "Want to Look Now?"

    Refer to the 👉 [Agents section](../tutorial/agents/index.md).

Like before, need a different agent? Just swap it out!

```python
from velora.models import LiquidPPO
import gymnasium as gym

env = gym.make('InvertedPendulum-v5')

model = LiquidPPO(4, 10, 1)
model.train(env, 128)
```

The `train()` method will create a [SQLite [:material-arrow-right-bottom:]](https://www.sqlite.org/) database called `metrics.db` in your local directory. This contains useful metrics that can be plotted to visualize the whole training process. How you use them is up to you!

We personally use and recommend a cloud-based solution (see the [Analytics Callbacks](../tutorial/callback.md#analytics) section) which uses these metrics automatically.

However, we've included this offline method separately just in case you prefer it! 😉 We'll talk more about these metrics later in the [Training Metrics](../tutorial/metrics.md) section.

## Making Predictions

For new predictions, we use the `predict()` method. This requires two parameters:

- `state` - the item to make a prediction on. Must be a `torch.Tensor`.
- `hidden` - the agent's hidden state.

Liquid Neural Networks are a recurrent architecture so a hidden state is required!

By default, we set `hidden` to `None`, so you don't need to provide it for a single prediction:

```python hl_lines="8"
from velora.models import LiquidDDPG
from velora.gym import add_core_env_wrappers

import gymnasium as gym


env = gym.make('InvertedPendulum-v5')
env = add_core_env_wrappers(env, device="cpu") # (1)

state, _ = env.reset()

model = LiquidDDPG(4, 10, 1)
action, hidden = model.predict(state)
```

1. Automatically turns environment states into `torch.Tensors`.

This code should work 'as is'.

Here, we get back an `action` prediction and an updated `hidden` state.

???+ Warning "Preparing the Environment"

    [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) environments are known to use `numpy` arrays for it's `state` spaces. Velora agents require `torch.Tensors` so you will need to pass the environment through a [NumpyToTorch [:material-arrow-right-bottom:]](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.NumpyToTorch) wrapper first.

    To make things easier, we strongly recommend you use the [`velora.gym.add_core_env_wrappers`](../tutorial/gym.md#core-wrappers) method instead. 

Things become slightly more complicated with multiple predictions because we need to feed the `hidden` state back into the `predict()` method like so:

```python
from velora.models import LiquidDDPG
from velora.gym import add_core_env_wrappers

import gymnasium as gym

env = gym.make('InvertedPendulum-v5')
env = add_core_env_wrappers(env, device="cpu")

model = LiquidDDPG(4, 10, 1)

hidden = None
state, _ = env.reset()

for ep in range(1000):
    action, hidden = model.predict(state, hidden)
    state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

    if done:
        break

env.close()
```

This code should work 'as is'.

Just like before, if you want to use a different agent, just swap out `LiquidDDPG` with another one. Glorious isn't it? 😉

---

That covers the basics! Next, we'll move onto `callbacks`. See you there! 👋
