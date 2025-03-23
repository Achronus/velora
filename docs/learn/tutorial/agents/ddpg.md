# Liquid DDPG

???+ api "API Docs"

    [`velora.models.LiquidDDPG`](../../reference/models/ddpg.md#velora.models.ddpg.LiquidDDPG)

Our first algorithm focuses on [Deep Deterministic Policy Gradients [:material-arrow-right-bottom:]](https://arxiv.org/abs/1509.02971) (DDPGs).

This algorithm is the first Deep RL architecture to use continuous action spaces.

It builds on techniques from [Deep Q-Networks [:material-arrow-right-bottom:]](https://www.nature.com/articles/nature14236) (DQNs), such as Replay Buffers and Target Networks and [DPGs [:material-arrow-right-bottom:]](https://proceedings.mlr.press/v32/silver14.pdf) for gradient updates.

To build one, we use the `LiquidDDPG` class.

## Building the Model

In it's simplest form, we can create one with just one line using three parameters:

- `state_dim` - the number of environment observation feature dimensions.
- `n_neurons` - the number of decision neurons in the network.
- `action_dim` - the number of actions the agent can take.

```python
from velora.models import LiquidDDPG

model = LiquidDDPG(4, 10, 1)
```

This code should work 'as is'.

This will create an instance of the model with the following default parameters:

- `optim=torch.optim.Adam` - using the Adam optimizer.
- `buffer_size=100_000` - a `ReplayBuffer` with a capacity of `100,000`.
- `actor_lr=1e-4` - the actor optimizer using a learning rate of `0.0001`.
- `critic_lr=1e-3` - the critic optimizer using a learning rate of `0.001`.
- `device=None` - no computation device set (E.g., `cpu`).

You can customize them freely using the required parameter name.

We strongly recommend that use the [`set_seed`](../../tutorial/utils.md#setting-a-seed) and [`set_device`](../../tutorial/utils.md#setting-a-device) utility methods before initializing the model to help with result reproducibility and faster training times:

```python
from velora.models import LiquidDDPG
from velora.utils import set_seed, set_device

set_seed(64)
device = set_device()

model = LiquidDDPG(4, 10, 1, device=device)
```

This code should work 'as is'.

## Training the Model

???+ api "API Docs"

    [`velora.models.LiquidDDPG.train(env, batch_size)`](../../reference/models/ddpg.md#velora.models.ddpg.LiquidDDPG.train)

Training the model is equally as simple! 😊

We just use the `train()` method given a `gymnasium.Env` and a `batch_size`:

```python hl_lines="9-10 13"
from velora.models import LiquidDDPG
from velora.utils import set_seed, set_device

import gymnasium as gym

set_seed(64)
device = set_device()

BATCH_SIZE = 128
env = gym.make("InvertedPendulum-v5")

model = LiquidDDPG(4, 10, 1, device=device)
model.train(env, BATCH_SIZE)
```

This code should work 'as is'.

This will train the agent with the following default parameters:

- `n_episodes=1000` - for `1000` episodes.
- `max_steps=1000` - with each episode having a maximum of `1000` steps.
- `noise_scale=0.3` - with an action exploration noise of `0.3`.
- `gamma=0.99` - a discount reward factor of `0.99`.
- `tau=0.005` - a soft update factor of `0.005`.
- `window_size=100` - a training progress status update every `100` episodes.

Like before, you can customize these freely using the required parameter name.

## Making a Prediction

???+ api "API Docs"

    [`velora.models.LiquidDDPG.predict(state, hidden)`](../../reference/models/ddpg.md#velora.models.ddpg.LiquidDDPG.predict)

To make a new prediction, we need to pass in a environment `state` and a `hidden` state.

```python
action, hidden = model.predict(state, hidden)
```

It comes with an optional `noise_scale` parameter (default is `0.1`) to experiment with the models robustness to noise.

Additional, it returns the `action` prediction and the `hidden` state.

If it's a one time prediction, `hidden=None` is perfect, but you'll likely be using this in a real-time setting so you'll need to pass the `hidden` state back into the next prediction.

???+ Warning "Preparing the Environment"

    [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) environments are known to use `numpy` arrays for it's `state` spaces. Velora agents require `torch.Tensors` so you will need to pass the environment through a [NumpyToTorch [:material-arrow-right-bottom:]](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.NumpyToTorch) wrapper first.

    To make things easier, we strongly recommend you use the [`velora.gym.add_core_env_wrappers`](../../tutorial/gym.md#core-wrappers) method instead.

Here's an example for usage in Notebooks ([Jupyter [:material-arrow-right-bottom:]](https://jupyter.org/) | [Google Colab [:material-arrow-right-bottom:]](https://colab.google/)):

```python
import gymnasium as gym
import matplotlib.pyplot as plt

from IPython.display import clear_output

from velora.models import LiquidDDPG
from velora.gym import add_core_env_wrappers
from velora.utils import set_seed, set_device

set_seed(64)
device = set_device()

env = gym.make("InvertedPendulum-v5", render_mode='rgb_array')
env = add_core_env_wrappers(env, device=device)

model = LiquidDDPG(4, 10, 1, device=device)
model.train(env, 128, n_episodes=100)

state, _ = env.reset()

episode_over = False
hidden = None

for _ in range(1000):
    action, hidden = model.predict(state, hidden)
    state, reward, terminated, truncated, _ = env.step(action)

    episode_over = terminated or truncated

    if episode_over:
        break

    clear_output(wait=True)
    plt.imshow(env.render())
    plt.show()

env.close()
```

This code should work 'as is'.
