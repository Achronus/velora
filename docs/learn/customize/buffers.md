# Working with Buffers

Buffers are a central piece for RL algorithms and are used heavily in our own implementations.

In Off-Policy agents we use a `ReplayBuffer` and in On-Policy, a `RolloutBuffer`.

???+ warning "Rollout Buffers"

    We've recently discontinued the `RolloutBuffer` and removed it from the framework due to instability issues with LNNs and on-policy agents.

    So, you'll only see the docs for the `ReplayBuffer` here!

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

### Add One Item

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

### Add Multiple Items

???+ api "API Docs"

    [`velora.buffer.ReplayBuffer.add_multi(state, action, reward, next_state, done)`](../reference/buffer.md#velora.buffer.BufferBase.add_multi)

Or, we can add multiple values at once using the `add_multi()` method. Like before, we can use a set of experience from a `Tuple` or the individual items.

The only difference, is that everything must be a `torch.Tensor`:

```python
import torch

exp = (
    torch.zeros((5, 4)),
    torch.ones((5, 1)),
    torch.ones((5, 1)),
    torch.zeros((5, 4)),
    torch.zeros(5, 1),
)

buffer.add_multi(*exp)
buffer.add_multi(
    torch.zeros((5, 4)),
    torch.ones((5, 1)),
    torch.ones((5, 1)),
    torch.zeros((5, 4)),
    torch.zeros(5, 1),
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

!!! warning

    We can only sample from the buffer after we have enough experience. This is dictated by your `batch_size`.

### Warming the Buffer

???+ api "API Docs"

    [`velora.buffer.ReplayBuffer.warm(agent, n_samples)`](../reference/buffer.md#velora.buffer.ReplayBuffer.warm)

Since the `ReplayBuffer` needs to have samples in it before we can `sample` from it. We can use a *warming* process to pre-populate the buffer.

We have a dedicated method for this called `warm()` that automatically gathers experience up to `n_samples` without affecting your `episode` count during training.

It requires two parameters:

| Parameter | Description | Example |
| --------- | ----------- | ------- |
| `agent`  | The Velora agent instance to generate samples with. | [`NeuroFlow`](../reference/models/nf.md) |
| `n_samples` | The number of samples to generate. | `batch_size * 2` |

And has one optional parameter:

| Parameter | Description            | Default            |
| --------- | ---------------------- | ------------------ |
| `num_envs`   | The number of vectorized environments to use for warming. | `8` |

```python
buffer.warm(agent, 1024)
```

### Check Size

Lastly, we can check the `current size` of the buffer:

```python
len(buffer)  # 1
```

### Full Replay Example

Here's a complete example of the code we've just seen:

```python
from velora.utils import set_device
from velora.models import NeuroFlow

import torch

device = set_device()

agent = NeuroFlow("CartPole-v1", 20, 128, device=device)

# Warm with at least 5 samples - can go over!
agent.buffer.warm(agent, 5, num_envs=2)

# Single experience
exp = (
    torch.zeros(agent.state_dim, device=device),
    torch.tensor((1.), device=device),
    2.,
    torch.zeros(agent.state_dim, device=device),
    False,
    torch.zeros(agent.actor.hidden_size, device=device),
)
agent.buffer.add(*exp)

# 3 more samples
exp = (
    torch.zeros((3, agent.state_dim), device=device),
    torch.ones((3, 1), device=device),
    torch.ones((3, 1), device=device),
    torch.zeros((3, agent.state_dim), device=device),
    torch.ones((3, 1), device=device),
    torch.zeros((3, agent.actor.hidden_size), device=device),
)
agent.buffer.add_multi(*exp)

# Get a batch
batch = agent.buffer.sample(batch_size=5)

len(agent.buffer)  # 10
```

This code should work 'as is'.

## Saving and Loading Buffers

Sometimes you might want to reuse a buffers state in a different project. Well, now you can!

We provide both a `save()` and `load()` feature for all buffers 😎.

### Saving

???+ api "API Docs"

    [`velora.buffer.BufferBase.save(dirpath)`](../reference/buffer.md#velora.buffer.BufferBase.save)

Once you've created a buffer and used it, simply pass in a `dirpath` to the `save` method. The final folder in the `dirpath` will be used to store the buffer's state. This includes:

- `buffer_metadata.json` - the buffers metadata
- `buffer_state.safetensors` - the buffers tensor state

You can change the filename prefix `buffer_` with the optional `prefix` parameter:

```python hl_lines="7"
from velora.buffer import ReplayBuffer

buffer = ReplayBuffer(100, 11, 3, 8, device="cpu")

buffer.save(
    'checkpoints/nf/CartPole_100', 
    prefix="buffer_" # (1)
)
```

1. Optional

This code should work 'as is'.

### Loading

???+ api "API Docs"

    [`velora.buffer.BufferBase.load(state_path, metadata)`](../reference/buffer.md#velora.buffer.BufferBase.load)

Then, to restore it into a new buffer instance, we use the `load()` method with the path to the `safetensors` file and the preloaded `metadata`:

```python
import json
from pathlib import Path
from velora.buffer import ReplayBuffer

root_path = 'checkpoints/nf/CartPole_100'

with Path(root_path, 'buffer_metadata.json').open("r") as f:
    metadata = json.load(f)

buffer = ReplayBuffer.load(Path(root_path, 'buffer_state'), metadata)
print(buffer.metadata())
# {
# 'capacity': 100,
# 'state_dim': 11,
# 'action_dim': 3,
# 'hidden_dim': 8,
# 'position': 0,
# 'size': 0,
# 'device': 'cpu'
# }
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
