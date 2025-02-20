# Working with Buffers

Buffers a central piece for RL algorithms and are used heavily in our own implementations.

In Off-Policy agents we use a `ReplayBuffer` and in On-Policy, a `RolloutBuffer`.

???+ api "API Docs"

    [`velora.buffer.ReplayBuffer`](../reference/buffer.md#velora.buffer.ReplayBuffer)

    [`velora.buffer.RolloutBuffer`](../reference/buffer.md#velora.buffer.RolloutBuffer)

We have our own implementations of these that are easy to work with 😊.

## Replay Buffer

To create a `ReplayBuffer`, simply give it a `capacity` and a `torch.device` (optional):

```python
from velora.buffer import ReplayBuffer
from velora.utils import set_device

device = set_device()
buffer = ReplayBuffer(capacity=100_000, device=device)
```

### Add Items

To add an item, we `push()` a set of `Experience` to it:

```python
from velora.buffer import Experience
import torch

exp = Experience(
    state=torch.zeros((1, 4)),
    action=1.,
    reward=2.,
    next_state=torch.zeros((1, 4)),
    done=False,
)

buffer.push(exp)
```

`Experience` is a simple dataclass that holds the information of a single environment `timestep`. We'll talk about them in more detail later.

### Get Samples

We can then `sample()` a batch of experience:

```python
batch = buffer.sample(batch_size=128)
```

This gives us a `BatchExperience` object. Like `Experience` we'll talk about that shortly, it's just another dataclass.

!!! note

    We can only sample from the buffer after we have enough experience. This is dictated by your `batch_size`.

### Check Size

Lastly, we can check the `current size` of the buffer:

```python
len(buffer)  # 1
```

### Full Replay Example

Here's a complete example of the code we've just seen:

```python
from velora.buffer import Experience, ReplayBuffer
from velora.utils import set_device

import torch

device = set_device()
buffer = ReplayBuffer(capacity=100_000, device=device)

exp = Experience(
    state=torch.zeros((1, 4)),
    action=1.,
    reward=2.,
    next_state=torch.zeros((1, 4)),
    done=False,
)

buffer.push(exp)

batch = buffer.sample(batch_size=1)

len(buffer)  # 1
```

This code should work 'as is'.

## Rollout Buffer

The `RolloutBuffer` is almost identical to the `ReplayBuffer` with the addition of an `empty()` method that must be used after the buffer is full.

To create one, give it a `capacity` and a `torch.device` (optional):

```python
from velora.buffer import RolloutBuffer
from velora.utils import set_device

device = set_device()
buffer = RolloutBuffer(capacity=10, device=device)
```

### Add Rollouts

To add an item, we `push()` a set of `Experience` to it:

```python
from velora.buffer import Experience
import torch

exp = Experience(
    state=torch.zeros((1, 4)),
    action=1.,
    reward=2.,
    next_state=torch.zeros((1, 4)),
    done=False,
)

buffer.push(exp)
```

Once the buffer is full, we need to `empty` it before we can add new samples:

```python
buffer.empty()
```

### Get All Samples

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
from velora.buffer import Experience, RolloutBuffer
from velora.utils import set_device

import torch

device = set_device()
buffer = RolloutBuffer(capacity=1, device=device)

exp = Experience(
    state=torch.zeros((1, 4)),
    action=1.,
    reward=2.,
    next_state=torch.zeros((1, 4)),
    done=False,
)

buffer.push(exp)
# buffer.push(exp)  # BufferError

batch = buffer.sample()

len(buffer)  # 1

buffer.empty()
len(buffer)  # 0
# buffer.sample()  # BufferError
```

This code should work 'as is'.

## Experience Dataclasses

As we've mentioned, `Experience` and `BatchExperience` are two dataclasses.

### Experience

`Experience` is the one you use to put data into the buffer:

```python
from typing import Tuple
from dataclasses import dataclass, astuple
import torch

@dataclass
class Experience:
    """
    Storage container for a single agent experience.

    Parameters:
        state (torch.Tensor): an environment observation
        action (float): agent action taken in the state
        reward (float): reward obtained for taking the action
        next_state (torch.Tensor): a newly generated environment observation
            after performing the action
        done (bool): environment completion status
    """

    state: torch.Tensor
    action: float
    reward: float
    next_state: torch.Tensor
    done: bool

    def __iter__(self) -> Tuple:
        return iter(astuple(self))
```

This has a unique iteration method that is useful for quickly unpacking a `List[Experience]`.

Here's an example:

```python
import random
from typing import List, Tuple

from velora.buffer import Experience

batch: List[Experience] = random.sample(buffer, batch_size)
states, actions, rewards, next_states, dones = zip(*batch)
```

This would give us a set of tuples, such as:

`((s1, s2, s3), (a1, a2, a3), (r1, r2, r3), (ns1, ns2, ns3), (d1, d2, d3))`

We could then convert these into tensors and store them inside `BatchExperience`. We won't go into the details of this, but it's useful to know it exists! 😉

#### Gymnasium Example

Here's an example of storing `Experience` into a `ReplayBuffer` using a Gymnasium environment:

```python
import gymnasium as gym
import torch
import numpy as np

from velora.buffer import ReplayBuffer, Experience
from velora.gym import add_core_env_wrappers


env = gym.make("InvertedPendulum-v5")
env = add_core_env_wrappers(env, device="cpu")
buffer = ReplayBuffer(capacity=100_000)

n_episodes = 10
max_steps = 1000
batch_size = 10
training_started = False
episode_rewards = []

print(f"{batch_size=}, getting buffer samples.")
for i_ep in range(n_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for i_step in range(max_steps):
        # Add network action prediction here...
        action = torch.tensor(env.action_space.sample())  # random action

        # Get a timestep of experience
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Add it to the buffer
        buffer.push(
            Experience(state, action.item(), reward, next_state, done),
        )

        # Check buffer is warmed
        if len(buffer) >= batch_size:
            if not training_started:
                print("Buffer warmed. Starting training...")
                training_started = True

            batch = buffer.sample(batch_size)

            # Network training code
            # ...

        # Reset state
        state = next_state
        episode_reward += reward


        # Terminate episode if done
        if done:
            break

    episode_rewards.append(reward)

    if training_started:
        avg_reward = np.mean(episode_rewards)

        print(
            f"Episode: {i_ep + 1}/{n_episodes}, "
            f"Reward: {avg_reward:.2f}"
        )

env.close()
```

This code should work 'as is'.

### BatchExperience

`BatchExperience` is the one you get out of the buffer:

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
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
```

All items in this class have the same shape `(batch_size, features)` and are easily accessible through their attributes, such as `batch.states`.

It's super convenient for doing calculations like this 😉:

```python
target_q = batch.rewards + (1 - batch.dones) * gamma * target_q
```

---

Next, we're going to talk about accessing existing models `Actor` and `Critic` classes 👋.
