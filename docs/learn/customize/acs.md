# Working with Existing Actor-Critic Classes

There might be a time where you want to experiment with specific `Actor` or `Critic` modules individually, without using the algorithm.

While it's very uncommon, it is possible with our framework!

## DDPG

???+ api "API Docs"

    [`velora.models.ddpg.DDPGActor`](../reference/models/ddpg.md#velora.models.ddpg.DDPGActor)

    [`velora.models.ddpg.DDPGCritic`](../reference/models/ddpg.md#velora.models.ddpg.DDPGCritic)

For DDPG, all you need to do is access the module directly, create your instance of the `DDPGActor` or `DDPGCritic` you want to use and then run the class like a normal PyTorch `nn.Module`:

```python
from velora.models.ddpg import DDPGActor, DDPGCritic
import torch

num_obs = 4
n_neurons = 10
num_actions = 1

actor = DDPGActor(4, 10, 1)
critic = DDPGCritic(4, 10, 1)

states = torch.zeros((1, 4))
actions = torch.ones((1, 1))

action_preds, actor_hidden = actor(states)
q_values, critic_hidden = critic(states, actions)
```

This code should run 'as is'.

Remember, all our algorithms are Liquid Networks, so they return a `prediction` and a `hidden` state! 😎

??? tip "Confused about inputs?"

    Calling a module directly removes the helpful tooltips for identify what parameters need to be passed in. There is a way around this!

    Start by using the `forward()` method and then remove it afterwards. While a little tedious, it gives you all the information you need 😊.

---

Next, we'll dive into working with static `backbones` that Velora offers 👋.
