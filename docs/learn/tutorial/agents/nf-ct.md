# NeuroFlow - Continuous

???+ api "API Docs"

    [`velora.models.NeuroFlowCT(env_id, actor_neurons, critic_neurons)`](../../reference/models/nf.md#velora.models.nf.NeuroFlowCT)

This algorithm focuses on the `continuous` action space implementation of `NeuroFlow` (NF).

???+ tip "Discrete vs. Continuous"

    The API and docs are identical to the `discrete` variant with some slight differences to the `init` parameters and the class name. We now use `NeuroFlowCT` instead of `NeuroFlow`.
    
    Feel free to jump to the part you need to save some time! 😀

It builds on top of a [Soft Actor-Critic (continuous) [:material-arrow-right-bottom:]](https://arxiv.org/abs/1801.01290) (SAC) base and combines a variety of well-known RL techniques with some of our own custom ones.

These include the following features:

- **Small Actor and large Critic networks** - from the paper: [Honey, I Shrunk The Actor: A Case Study on Preserving Performance with Smaller Actors in Actor-Critic RL  [:material-arrow-right-bottom:]](https://arxiv.org/abs/2102.11893).
- **Differing Actor-Critic architectures** - the Actor uses a [`LiquidNCPNetwork`](../../reference/models/lnn.md#velora.models.lnn.ncp.LiquidNCPNetwork) and the Critic's use [`NCPNetworks`](../../reference/models/lnn.md#velora.models.lnn.ncp.NCPNetwork).
- **Automatic Entropy Adjustment (Learned)** - from the paper: [Soft Actor-Critic Algorithms and Applications [:material-arrow-right-bottom:]](https://arxiv.org/abs/1812.05905).

Plus more, coming soon.

!!! warning "Agent's Future"

    Cyber environments typically use `discrete` action spaces so this agent won't be used as heavily as the `discrete` variant.

    As such, this may be discontinued and removed in a future release. Alternatively, we may still keep it but limit it to it's current features without any additional ones that are planned for the full algorithm.

To build one, we use the `NeuroFlowCT` class.

## Building the Model

In it's simplest form, we can create one with just one line using three parameters:

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| `env_id`  | The Gymnasium environment ID. | `InvertedPendulum-v5` |
| `actor_neurons` | The number of decision/hidden nodes for the Actor network. | `20` or `40` |
| `critic_neurons` | The number of decision/hidden nodes for the Critic networks. We recommend this to be higher than the Actor network | `128` or `256` |

```python
from velora.models import NeuroFlowCT

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128)
```

This code should work 'as is'.

### Optional Parameters

This will create an instance of the model with the following default parameters:

| Parameter | Description            | Default            |
| --------- | ---------------------- | ------------------ |
| `optim`   | The PyTorch optimizer. | `torch.optim.Adam` |
| `buffer_size` | The `ReplayBuffer` size. | `1M` |
| `actor_lr` | The actor optimizer learning rate. | `0.0003` |
| `critic_lr` | The critic optimizer learning rate. | `0.0003` |
| `alpha_lr` | The entropy optimizer learning rate. | `0.0003` |
| `initial_alpha` | The starting entropy coefficient. | `1.0` |
| `log_std` | The `(low, high)` bounds for the log standard deviation of the action distribution. Used to control the variance of actions. | `(-5, 2)` |
| `tau` | The soft update factor for slowly updating the target network weights. | `0.005` |
| `gamma` | The reward discount factor. | `0.99` |
| `device` | The device to perform computations on. E.g., `cpu` or `cuda:0`. | `None` |
| `seed` | The random generation seed for `Python`, `PyTorch`, `NumPy` and `Gymnasium`. When `None`, seed is automatically generated. | `None` |

You can customize them freely using the required parameter name.

We strongly recommend that use the [`set_device`](../../tutorial/utils.md#setting-a-device) utility method before initializing the model to help with faster training times:

```python
from velora.models import NeuroFlowCT
from velora.utils import set_device

device = set_device()

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128, device=device)
```

This code should work 'as is'.

`NeuroFlowCT` uses the [`set_seed`](../../tutorial/utils.md#setting-a-seed) utility method automatically when the model's `seed=None`. This saves you having to manually create it first! 😉

## Training the Model

???+ api "API Docs"

    [`velora.models.NeuroFlowCT.train(batch_size)`](../../reference/models/nf.md#velora.models.nf.NeuroFlowCT.train)

Training the model is equally as simple! 😊

We just use the `train()` method given a `batch_size`:

```python hl_lines="7"
from velora.models import NeuroFlowCT
from velora.utils import set_device

device = set_device()

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128, device=device)
model.train(256)
```

This code should work 'as is'.

### Optional Parameters

This will train the agent with the following default parameters:

| Parameter | Description            | Default            |
| --------- | ---------------------- | ------------------ |
| `n_episodes` | The number of episodes to train for. | `10k` |
| `callbacks` | A list of training callbacks applied during the training process. | `None` |
| `log_freq` | The metric logging frequency for offline and online analytics (in episodes). | `10` |
| `display_count` | The console training progress frequency (in episodes). | `100` |
| `window_size` | The reward moving average size (in episodes). | `100` |
| `max_steps` | The total number of steps per episode. | `1000` |
| `warmup_steps` | The number of samples to generate in the buffer before starting training. If `None` uses `batch_size * 2`. | `None` |

Like before, you can customize these freely using the required parameter name.

## Making a Prediction

???+ api "API Docs"

    [`velora.models.NeuroFlowCT.predict(state, hidden)`](../../reference/models/nf.md#velora.models.nf.NeuroFlowCT.predict)

To make a new prediction, we need to pass in a environment `state` and a `hidden` state.

```python
action, hidden = model.predict(state, hidden)
```

### Optional Parameters

This will make a prediction with the following default parameters:

| Parameter | Description            | Default            |
| --------- | ---------------------- | ------------------ |
| `train_mode` | A flag for swapping between *deterministic* and *stochastic* action predictions. <ul><li>When `False` - deterministic action predictions. Recommend for evaluating the model.</li><li>When `True` - stochastic action predictions. Required for training the model.</li></ul> | `False` |

Every prediction returns the `action` prediction and the `hidden` state.

If it's a one time prediction, `hidden=None` is perfect, but you'll likely be using this in a real-time setting so you'll need to pass the `hidden` state back into the next prediction and use a pre-wrapped environment (`model.eval_env`).

### Example

Here's a code example:

```python
from velora.models import NeuroFlowCT
from velora.utils import set_device

device = set_device()

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128, device=device, seed=64)
model.train(128, n_episodes=100)

# Set prediction env
env = model.eval_env

# Run trained agent for 5 episodes
ep_total = 5
for i_ep in range(1, ep_total + 1):
    state, _ = env.reset()
    hidden = None

    while True:
        action, hidden = model.predict(state, hidden)
        state, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

        if episode_over:
            ep_return = info["episode"]["r"].item()
            print(f"Episode: {i_ep}/{ep_total}, Reward: {ep_return:.2f}")
            break
```

This code should work 'as is'.

---

Now, let's see how we can use our `offline` training metrics! 🚀
