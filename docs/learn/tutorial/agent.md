# Agent Basics

RL agents are at the heart of Velora's framework and are the fastest way to get started with experiments.

Each agent has subtle differences but we've designed them to act like drop-in replacements of each other. Their underlying functionality may differ, but their core API is identical with the exception of optional hyperparameters.

At it's core, you have three main operations:

- **Initialize** - creation of the model.
- **Training** - running the model on an environment.
- **Prediction** - making predictions on unseen data.

## Creating Agents

Creating a agent is really easy! We simply declare our agent from the `velora.models` API and create a class instance.

Each model requires three main parameters:

1. `env_id` - the Gymnasium environment ID. E.g., `CartPole-v1` or `InvertedPendulum-v5`.
2. `actor_neurons` - the number of decision/hidden nodes for the Actor network (e.g., `20` or `40`).
3. `critic_neurons` - the number of decision/hidden nodes for the Critic networks. We recommend this to be higher (`128` or `256`) than the Actor network.

And that's it! Here's an example:

```python
from velora.models import NeuroFlowCT

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128)
```

This code should work 'as is'.

Want to use a different agent? Just swap out `NeuroFlowCT` with a different one!

```python
from velora.models import NeuroFlow

model = NeuroFlow("CartPole-v1", 20, 128)
```

It really is that easy! 🤩

???+ note "Agent Parameters"

    Each agent comes with a set of optional parameters that can be customized. 
    
    You can read more about them in the [`Agents`](../tutorial/agents/index.md) documentation section.

## Training an Agent

Training an agent is just as easy!

We use the `train` method, supply it with a `batch_size` and boom 💥, your agent will start training for `1000` episodes:

```python
from velora.models import NeuroFlowCT

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128)
model.train(128)
```

This code should work 'as is'.

Want to change the number of episodes? Use the `n_episodes` parameter! What about the console logged training status frequency? Use the `display_count` parameter!

```python
model.train(env, 128, n_episodes=10_000, display_count=10)
```

These are only two of the optional parameters for the `train()` method. Another worth mentioning is [`callbacks`](../tutorial/callback.md) but we'll talk about them later.

Like before, need a different agent? Just swap it out!

```python
from velora.models import NeuroFlow

model = NeuroFlow("CartPole-v1", 20, 128)
model.train(64)
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

```python
from velora.models import NeuroFlowCT

# Set prediction environment
env = model.eval_env

state, _ = env.reset()

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128)
action, hidden = model.predict(state)
```

This code should work 'as is'.

Here, we get back an `action` prediction and an updated `hidden` state.

Things become slightly more complicated with multiple predictions because we need to feed the `hidden` state back into the `predict()` method like so:

```python
from velora.models import NeuroFlowCT

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128)

# Set prediction environment
env = model.eval_env

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

Just like before, if you want to use a different agent, just swap out `NeuroFlowCT` with another one. Glorious isn't it? 😉

---

That covers the basics! Next, we'll move onto `callbacks`. See you there! 👋
