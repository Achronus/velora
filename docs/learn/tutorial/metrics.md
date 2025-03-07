# Working with Training Metrics

???+ api "API Docs"

    [`velora.training.TrainMetrics`](../reference/training.md#velora.training.TrainMetrics)

Understanding how your agent is learning is extremely important for figuring out how to optimize its performance and also fix it when it's broken.

To do this offline, we use a *utility* class called `TrainMetrics` that is returned after an agent has completed it's training cycle using the `train()` method.

???+ warning "Metrics are a one time use unless you manually save them"

    Normally, agents are trained for thousands of episodes and saving metrics can take up a lot of storage space. So, we **DO NOT** save the metrics internally inside the agent even if it takes *days* or *weeks* to train it.

    If you want to access them again, you will need to manually save them using the [`save()`](#saving-metrics) method.

    ❗ We highly recommend you use a cloud-based solution instead (see the [Analytics Callbacks](../tutorial/callback.md#analytics) section) but understand that it's not for everyone. 

## Storage

The most notable feature of the `TrainMetrics` class is the `storage` attribute.

This contains unique storage containers for each of the episodic statistics as `MovingMetric` dataclasses. We use these to calculate the moving averages in real-time using a window size for the `mean` and `std`.

Typically, you won't use this yourself. Instead, you'll want to pull the episodic values using the respective `attributes`.

These include:

- `ep_rewards` - a `List[float]` of episode rewards.
- `ep_lengths` - a `List[int]` of the number of steps taken per episode.
- `actor_losses` - a `List[float]` of Actor loss values.
- `critic_losses` - a `List[float]` of Critic loss values.

```python
from velora.models import [model]

import gymnasium as gym

env = # ...
model = # ...

metrics = model.train(env, 128)

metrics.ep_rewards  # [10., 2., 6., 8., 9.]
metrics.ep_lengths  # [20, 10, 30, 40, 12]
```

## Saving Metrics

???+ api "API Docs"

    [`velora.training.TrainMetrics.save(filepath)`](../reference/training.md#velora.training.TrainMetrics.save)

Manually saving the metrics is really easy. We just use the `save()` method with a given `filepath`! 😊

```python
metrics.save('metrics/my_metrics.pt')
```

## Loading Metrics

???+ api "API Docs"

    [`velora.training.SimpleMetricStorage.load(filepath)`](../reference/training.md#velora.training.SimpleMetricStorage.load)

Loading metrics is also easy, but instead of using the `TrainMetrics` class, we use the `SimpleMetricStorage` class instead.

```python
from velora.training import SimpleMetricStorage

metrics = SimpleMetricStorage.load('metrics/my_metrics.pt')

metrics.ep_rewards  # [10., 2., 6., 8., 9.]
metrics.ep_lengths  # [20, 10, 30, 40, 12]
```

??? question "Why a different container?"

    The `TrainMetrics.storage` container uses attributes as `MovingMetric` dataclasses. These are used for real-time calculations and are not needed in offline settings, so a lot of the functionality is redundant when loading a set of saved metrics.

    To reduce the memory footprint, we use a lightweight and simplified storage container that only has `attributes` and a `load` method. 

It includes all the same `attributes` mentioned in the [Storage](#storage) section just in a lightweight storage container.

---

Next, we'll dive into the utility methods for `Gymnasium` 👋.
