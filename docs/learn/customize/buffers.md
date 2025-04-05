# Working with Buffers

Buffers are a central piece for RL algorithms and are used heavily in our own implementations.

In Off-Policy agents we use a `ReplayBuffer` and in On-Policy, a `RolloutBuffer`.

We have our own implementations of these that are easy to work with 😊.

## Replay Buffer

???+ api "API Docs"

    [`velora.buffer.ReplayBuffer(capacity, state_dim, action_dim)`](../reference/buffer.md#velora.buffer.ReplayBuffer)

To create a `ReplayBuffer`, simply give it a `capacity`, `state_dim`, `action_dim`, `hidden_dim` and a `torch.device` (optional):

```python
from velora.buffer import ReplayBuffer
from velora.utils import set_device

device = set_device()
buffer = ReplayBuffer(
    capacity=100_000, 
    state_dim=11, 
    action_dim=3,
    hidden_dim=8, 
    device=device
)
```

### Add Items

???+ api "API Docs"

    [`velora.buffer.ReplayBuffer.add(state, action, reward, next_state, done)`](../reference/buffer.md#velora.buffer.BufferBase.add)

To add an item, we use the `add()` method with a set of experience from a `Tuple` or the individual items:

```python
import torch

exp = (
    torch.zeros((1, 4)),
    torch.tensor((1.)),
    2.,
    torch.zeros((1, 4)),
    False,
)

buffer.add(*exp)
buffer.add(
    torch.zeros((1, 4)),
    torch.tensor((1.)),
    2.,
    torch.zeros((1, 4)),
    False,
)
```

### Get Samples

???+ api "API Docs"

    [`velora.buffer.ReplayBuffer.sample(batch_size)`](../reference/buffer.md#velora.buffer.ReplayBuffer.sample)

We can then `sample()` a batch of experience:

```python
batch = buffer.sample(batch_size=128)
```

This gives us a `BatchExperience` object. We'll talk about this later.

!!! note

    We can only sample from the buffer after we have enough experience. This is dictated by your `batch_size`.

### Warming the Buffer

???+ api "API Docs"

    [`velora.buffer.ReplayBuffer.warm(agent, env_name, n_samples)`](../reference/buffer.md#velora.buffer.ReplayBuffer.warm)

The `ReplayBuffer` needs to have samples in it before we can `sample` from it. We call this the *warming* process.

We have a dedicated method for this called `warm()` that automatically gathers experience up to `n_samples` without effecting your `episode` count during training.

It takes three parameters:

- `agent` - the `RLAgent` instance to generate samples with. E.g., [`LiquidDDPG`](../reference/models/ddpg.md)
- `env_name` - the [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) environment name ID (`env.spec.id`). E.g., `InvertedPendulum-v5`.
- `n_samples` - the number of samples to generate. E.g., the `batch_size`

```python
buffer.warm(agent, env.spec.id, 128)
```

### Check Size

Lastly, we can check the `current size` of the buffer:

```python
len(buffer)  # 1
```

### Full Replay Example

Here's a complete example of the code we've just seen:

```python
from velora.buffer import ReplayBuffer
from velora.utils import set_device
from velora.models.ddpg import LiquidDDPG

import gymnasium as gym
import torch

device = set_device()

env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = LiquidDDPG(state_dim, 10, action_dim, device=device)
buffer = ReplayBuffer(
    100_000, 
    state_dim, 
    action_dim, 
    agent.actor.ncp.hidden_size,
    device=device
)

# Warm with 5 samples
buffer.warm(agent, env.spec.id, 5)

# Single experience
exp = (
    torch.zeros(state_dim, device=device),
    torch.tensor((1.), device=device),
    2.,
    torch.zeros(state_dim, device=device),
    False,
)
buffer.add(*exp)

# Get a batch
batch = buffer.sample(batch_size=5)

len(buffer)  # 6
```

This code should work 'as is'.

## Rollout Buffer

???+ api "API Docs"

    [`velora.buffer.RolloutBuffer(capacity, state_dim, action_dim)`](../reference/buffer.md#velora.buffer.RolloutBuffer)

The `RolloutBuffer` is almost identical to the `ReplayBuffer` with the addition of an `empty()` method that must be used after the buffer is full.

To create one, give it a `capacity`, `state_dim`, `action_dim`, `hidden_dim` and a `torch.device` (optional):

```python
from velora.buffer import RolloutBuffer
from velora.utils import set_device

device = set_device()
buffer = RolloutBuffer(
    capacity=10, 
    state_dim=11,
    action_dim=3,
    hidden_dim=8,
    device=device
)
```

### Add and Empty Rollouts

???+ api "API Docs"

    [`velora.buffer.RolloutBuffer.add(state, action, reward, next_state, done)`](../reference/buffer.md#velora.buffer.BufferBase.add)

    [`velora.buffer.RolloutBuffer.empty()`](../reference/buffer.md#velora.buffer.RolloutBuffer.empty)

To add an item, we use the `add()` method with a set of experience from a `Tuple` or the individual items:

```python
import torch

exp = (
    torch.zeros((1, 4)),
    torch.tensor((1.)),
    2.,
    torch.zeros((1, 4)),
    False,
)

buffer.add(*exp)
buffer.add(
    torch.zeros((1, 4)),
    torch.tensor((1.)),
    2.,
    torch.zeros((1, 4)),
    False,
)
```

Once the buffer is full, we need to `empty()` it before we can add new samples:

```python
buffer.empty()
```

### Get All Samples

???+ api "API Docs"

    [`velora.buffer.RolloutBuffer.sample()`](../reference/buffer.md#velora.buffer.RolloutBuffer.sample)

We can get the complete experience from the rollout buffer using the `sample()` method:

```python
rollouts = buffer.sample()
```

This gives us a `BatchExperience` object.

### Check Buffer Size

Lastly, we can check the `current size` of the buffer:

```python
len(buffer)  # 1
```

### Full Rollout Example

Here's a complete example of the code we've just seen:

```python
from velora.buffer import RolloutBuffer
from velora.utils import set_device

import torch

device = set_device()
buffer = RolloutBuffer(1, 4, 1, 8, device=device)

exp = (
    torch.zeros((1, 4)),
    torch.tensor((1.)),
    2.,
    torch.zeros((1, 4)),
    False,
)

buffer.add(*exp)
# buffer.add(*exp)  # BufferError

batch = buffer.sample()

len(buffer)  # 1

buffer.empty()
len(buffer)  # 0
# buffer.sample()  # BufferError
```

This code should work 'as is'.

## Saving and Loading Buffers

???+ api "API Docs"

    [`velora.buffer.BufferBase.save(filepath)`](../reference/buffer.md#velora.buffer.BufferBase.save)

    [`velora.buffer.BufferBase.load(filepath)`](../reference/buffer.md#velora.buffer.BufferBase.load)

Sometimes you might want to reuse a buffers state in a different project. Well, now you can!

We provide both a `save()` and `load()` feature for all buffers 😎.

Once you've created a buffer and used it, simply pass in a `filepath` to the `save` method to store it like `PyTorch` parameters:

```python
from velora.buffer import ReplayBuffer

buffer = ReplayBuffer(100, 11, 3, device="cpu")

buffer.save('checkpoints/buffer_100_cpu.pt')
```

This code should work 'as is'.

Then, to restore it into a new buffer instance, we use `load()` like so:

```python
from velora.buffer import ReplayBuffer

buffer = ReplayBuffer.load('checkpoints/buffer_100_cpu.pt')
```

This code should work 'as is'.

## BatchExperience

???+ api "API Docs"

    [`velora.buffer.BatchExperience`](../reference/buffer.md#velora.buffer.BatchExperience)

Earlier, we mentioned the `BatchExperience` object. This is a dataclass that stores ours experience as separate tensors and allows you to easily extract them using their attributes.

As mentioned, `BatchExperience` is the one you get out of the buffer:

```python
from dataclasses import dataclass
import torch

@dataclass
class BatchExperience:
    """
    Storage container for a batch agent experiences.

    Parameters:
        states (torch.Tensor): a batch of environment observations
        actions (torch.Tensor): a batch of agent actions taken in the states
        rewards (torch.Tensor): a batch of rewards obtained for taking the actions
        next_states (torch.Tensor): a batch of newly generated environment
            observations following the actions taken
        dones (torch.Tensor): a batch of environment completion statuses
        hiddens (torch.Tensor): a batch of prediction network hidden states
            (e.g., Actor)
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    hiddens: torch.Tensor
```

All items in this class have the same shape `(batch_size, features)` and are easily accessible through their attributes, such as `batch.states`.

It's super convenient for doing calculations like this 😉:

```python
target_q = batch.rewards + (1 - batch.dones) * gamma * target_q
```

---

Next, we're going to talk about accessing existing models `Actor` and `Critic` classes 👋.
