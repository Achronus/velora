# Gymnasium Utility Methods

Velora uses the [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) package as it's environment provider, which you'll use to train your agents and view their performance.

Most of the time, you will interact with their API like normal, using the `make()` method:

```python
import gymnasium as gym

env = gym.make("InvertedPendulum-v5")
```

However, sometimes you'll need to expand their existing functionality using [gymnasium.Wrappers [:material-arrow-right-bottom:]](https://gymnasium.farama.org/api/wrappers/table/).

Velora offers two simple methods to help with this: `wrap_gym_env` and `add_core_env_wrappers`.

## Wrapping Gymnasium Environments

???+ api "API Docs"

    [`velora.gym.wrap_gym_env`](../reference/gym.md#velora.gym.wrap_gym_env)

`wrap_gym_env` is a quick way to create new environments that with wrappers automatically applied. Normally, you'd have to apply wrappers, one by one like this:

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

    [`velora.gym.add_core_env_wrappers`](../reference/gym.md#velora.gym.add_core_env_wrappers)

We've also added a `add_core_env_wrappers` method that applies specific wrappers that are required by every prebuilt algorithm in Velora.

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

You typically won't need to use this yourself, but it's useful to know for your own builds! 😊

## Finding Environments

???+ api "API Docs"

    [`velora.gym.EnvSearch`](../reference/gym.md#velora.gym.EnvSearch)

    [`velora.gym.EnvResult`](../reference/gym.md#velora.gym.EnvResult)

Sometimes you may want to quickly find a specific environment and use the latest version of it without searching through the [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) documentation.

We've added a utility class to help with this called `EnvSearch`.

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
