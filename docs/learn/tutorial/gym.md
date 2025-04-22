# Gymnasium Utility Methods

Velora uses the [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) package as it's main environment provider, which is used to train agents and view their performance.

Normally, you would use the `make()` method to create an environment, like so:

```python
import gymnasium as gym

env = gym.make("InvertedPendulum-v5")
```

We've removed the need to do this to simplify the implementation and to allow us to easily integrate the environment with our agents.

Instead, now we pass in the `env_id` to the agent of our choice:

```python
from velora.models import NeuroFlow

model = NeuroFlow("InvertedPendulum-v5", 20, 128)
```

Unfortunately, this takes away a lot of your freedom for adding custom wrappers and we are looking at ways to incorporate this in a future release. For now, we want to keep things simple while we fully flesh out the agents implementation details.

## Utility Methods

Under rare circumstances you might want to use a Gymnasium environment to quickly explore it before using a Velora agent.

To help with this, we've added some utility methods that you might find useful.

### Wrapping Gymnasium Environments

???+ api "API Docs"

    [`velora.gym.wrap_gym_env(env, wrappers)`](../reference/gym.md#velora.gym.wrap_gym_env)

`wrap_gym_env` is a quick way to create new environments with wrappers automatically applied. Normally, you'd have to apply wrappers, one by one like this:

```python
import gymnasium as gym

from gymnasium.wrappers import NormalizeObservation, NormalizeReward

env = gym.make("InvertedPendulum-v5")
env = NormalizeObservation(env, epsilon=1e-8)
env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
```

It's pretty tedious, so we've simplified it a little with the `wrap_gym_env` method:

```python
from functools import partial

from gymnasium.wrappers import (
    NormalizeObservation, 
    NormalizeReward, 
    RecordEpisodeStatistics,
)

from velora.gym import wrap_gym_env

env = wrap_gym_env("InvertedPendulum-v5", [
    partial(NormalizeObservation, epsilon=1e-8),
    partial(NormalizeReward, gamma=0.99, epsilon=1e-8),
    RecordEpisodeStatistics,
])
```

This code should work 'as is'.

Now, you just supply the environment `name` and a of `List[gym.Wrappers]` or `List[partial]` wrappers and your environment is good to go! 😎

### Core Wrappers

???+ api "API Docs"

    [`velora.gym.add_core_env_wrappers(env, device)`](../reference/gym.md#velora.gym.add_core_env_wrappers)

We've also added a `add_core_env_wrappers` method that applies specific wrappers required by every Velora agent.

It applies the following wrappers (in order):

- [RecordEpisodeStatistics [:material-arrow-right-bottom:]](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordEpisodeStatistics) - for easily retrieving episode statistics.
- [NumpyToTorch [:material-arrow-right-bottom:]](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.NumpyToTorch) - for turning environment feedback into `PyTorch` tensors.

Here's an example:

```python
from functools import partial

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

from velora.gym import wrap_gym_env, add_core_env_wrappers

env = wrap_gym_env("InvertedPendulum-v5", [
    partial(NormalizeObservation, epsilon=1e-8),
    partial(NormalizeReward, gamma=0.99, epsilon=1e-8),
])
env = add_core_env_wrappers(env, device="cpu")

# Or ..
env = gym.make("InvertedPendulum-v5")
env = add_core_env_wrappers(env, device="cpu")
```

In previous versions, you would manually need to use this yourself when working with the `predict()` method to quickly convert the environment from `numpy` arrays to `torch` tensors.

We found this to be very tedious and quite confusing in some instances, so instead, we've simplified this process by adding a `.eval_env` attribute to every agent to remove this process:

```python
from velora.models import NeuroFlowCT

model = NeuroFlowCT("InvertedPendulum-v5", 20, 128)
env = model.eval_env

# 👆 Equivalent to ..
import gymnasium as gym

env = gym.make("InvertedPendulum-v5")
env = add_core_env_wrappers(env, device="cpu")
```

You can see an example of this in the [Agent Basics - Making Predictions](../tutorial/agent.md#making-predictions) section.

## Finding Environments

???+ api "API Docs"

    [`velora.gym.EnvSearch`](../reference/gym.md#velora.gym.EnvSearch)

    [`velora.gym.EnvResult`](../reference/gym.md#velora.gym.EnvResult)

We've also added a unique approach to quickly finding a specific environment at it's latest version without having to search through the [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) documentation.

You can do this with a utility class called `EnvSearch`.

Every method attached to the class returns a `List[EnvResult]` where `EnvResult` are objects that have a `name` and `type` to help you quickly determine which environment fits your use case.

There are three search methods split into two categories:

- Searching for a specific environment
- Getting a list of available environments by type

### Specific One

???+ api "API Docs"

    [`velora.gym.EnvSearch.find(query)`](../reference/gym.md#velora.gym.EnvSearch.find)

Let's say you want to quickly find the latest `LunarLander` environment, specifically the `continuous` one.

You can use the `find()` method:

```python
import gymnasium as gym

from velora.gym import EnvSearch


result = EnvSearch.find('LunarLander')
# [
#   EnvResult(name='LunarLander-v3', type='discrete'), 
#   EnvResult(name='LunarLanderContinuous-v3', type='continuous')
# ]

result2 = EnvSearch.find('Pendulum')
# [
#    EnvResult(name='Pendulum-v1', type='continuous'),
#    EnvResult(name='InvertedPendulum-v5', type='continuous'),
#    EnvResult(name='InvertedDoublePendulum-v5', type='continuous')
# ]

# Quick usage with the Gymnasium API
name = result[-1].name # 'LunarLanderContinuous-v3'
env = gym.make(name)
```

This code should work 'as is'.

Given a `query` string (part of the name or the full name), it will give you a list of relevant results.

### By Type

???+ api "API Docs"

    [`velora.gym.EnvSearch.discrete()`](../reference/gym.md#velora.gym.EnvSearch.discrete)

    [`velora.gym.EnvSearch.continuous()`](../reference/gym.md#velora.gym.EnvSearch.continuous)

Not sure what environments are available? Looking for your next `discrete` one or `continuous` one? We've got you covered! 😉

Simply use the `discrete()` or `continuous()` methods to get a complete list of available options:

=== "Discrete"

    ```python
    from velora.gym import EnvSearch

    results = EnvSearch.discrete()
    # [
    #   EnvResult(name='CartPole-v1', type='discrete'),
    #   # ...
    # ]
    ```

=== "Continuous"

    ```python
    from velora.gym import EnvSearch

    results = EnvSearch.continuous()
    # [
    #   EnvResult(name='MountainCarContinuous-v0', type='continuous'),
    #   # ...
    # ]
    ```

This code should work 'as is'.

???+ warning

    Environments are split into two categories `discrete` or `continuous` based on their `action_space`:
    
    - `discrete` requires a `gym.spaces.Discrete` space
    - `continuous` requires a `gym.spaces.Box` space

    Any other `action_space` type is ignored.

---

Up next, we'll take a look at the generic utility methods Velora has to offer 👋.
