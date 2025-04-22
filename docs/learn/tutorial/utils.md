# General Utility Methods

Velora has a couple of utility methods to simplify minor, but tedious operations.

We've split these into two main categories:

- `core` - ones you'll use often.
- `torch` - extras focusing on `PyTorch` operations.

## Core

When you start a new experiment there are two things you will often ALWAYS want to do: set a random `seed` and initialize your computation `device`.

We've got two methods that help with this: `set_seed` and `set_device`.

???+ tip

    We highly recommend you call these two methods at the `start` of your experiments. Setting a seed helps with reproducibility and setting a device reduces training time.

    Both are extremely useful and shouldn't be ignored.

### Setting a Seed

???+ api "API Docs"

    [`velora.utils.set_seed()`](../reference/utils.md#velora.utils.core.set_seed)

Setting a seed controls the randomness for `Python`, `PyTorch` and `NumPy`, making your experiment results consistent and reproducible.

You have two options here:

1. Pass your own seed value
2. Let it generate one automatically

```python
from velora.utils import set_seed

seed = set_seed(64)
print(seed)  # 64

seed = set_seed()
print(seed)  # A random seed
```

This code should work 'as is'.

Then, you would pass the `seed` to the `seed` parameter of to any of Velora's agents `init` method.

### Setting a Device

???+ api "API Docs"

    [`velora.utils.set_device(device)`](../reference/utils.md#velora.utils.core.set_device)

Setting a device controls where your `PyTorch` computations are performed.

You can pass a device name in manually, or you can leave it blank and the device will be assigned automatically to `cuda:0` (if GPU enabled) or `cpu` (without).

```python
from velora.utils import set_device

# Automatically assigned
device = set_device()  # torch.device('cuda:0') | torch.device('cpu')

# Static assignment
device = set_device("cuda:1")  # torch.device('cuda:1')
```

This code should work 'as is'.

Then, you would pass the `device` to the `device` parameter to any of Velora's agents `init` method.

## Torch

During your own experiments, you might find yourself in need of a quick way to convert data to a `PyTorch` tensor, or want to perform a parameter update between two networks, or even quickly check the number of `parameters` in a model.

We've got a few methods to help with this:

- `to_tensor` - converts a list of data to a `torch.Tensor`.
- `stack_tensor` - stacks a list of `torch.Tensors` into a single one.
- `soft_update` - performs a soft parameter update between two `torch.nn.Modules`.
- `hard_update` - performs a hard parameter update between two `torch.nn.Modules`.
- `total_parameters` - calculates the total number of parameters for a `torch.nn.Module`.
- `active_parameters` - calculates the number of active parameters for a `torch.nn.Module`.

### Item List to Tensor

???+ api "API Docs"

    [`velora.utils.to_tensor(items)`](../reference/utils.md#velora.utils.torch.to_tensor)

Let's say you have a `List[int]` that are action values and you want to load them onto your GPU quickly while maintaining the data type.

We can quickly do this using `to_tensor`:

```python
from velora.utils import to_tensor, set_device
import torch

actions = [1, 2, 4, 5, 1, 1]

# Set our device
device = set_device()

# Convert the actions to a tensor
actions = to_tensor(actions, dtype=torch.int, device=device)
```

This code should work 'as is'.

By default, `dtype=torch.float32` and `device=None` so if you had a set of reward values (`float`) you'd only need to set `device`.

```python
from velora.utils import to_tensor, set_device

rewards = [1., 5, -1., -10., 1., 1.]

# Set our device
device = set_device()

# Convert the rewards to a tensor with default `dtype`
rewards = to_tensor(rewards, device=device)
```

This code should work 'as is'.

### Stacking Tensors

???+ api "API Docs"

    [`velora.utils.stack_tensor(items)`](../reference/utils.md#velora.utils.torch.stack_tensor)

What if we wanted to merge two `torch.Tensors` together? A common example would be environment observations in a buffer.

For our simple example, we'll merge the `actions` and `rewards` from our previous section:

```python
from velora.utils import stack_tensor, set_device
import torch

actions = torch.tensor([1, 2, 4, 5, 1, 1], dtype=torch.int)
rewards = torch.tensor([1., 5, -1., -10., 1., 1.])

# Set our device
device = set_device()

# stack tensors with default `dtype`
values = stack_tensor([actions, rewards], device=device)  # (2, 6)
```

This code should work 'as is'.

Like [`to_tensor`](#item-list-to-tensor), we use `dtype=torch.float32` and `device=None` as defaults.

### Soft Network Parameter Updates

???+ api "API Docs"

    [`velora.utils.soft_update(source, target, tau)`](../reference/utils.md#velora.utils.torch.soft_update)

Some algorithms, like [`NeuroFlow`](../tutorial/agents/nf.md), perform soft target parameter updates using a hyperparameter $\tau$.

This method performs that exact process. Given a `source` network, `target` network, and soft update factor `tau`, we iterate through our parameters and perform a soft update:

```python
from copy import deepcopy

from velora.models.sac.discrete import SACActorDiscrete
from velora.utils import soft_update

# Set two networks
actor = SACActorDiscrete(4, 10, 1)
target = deepcopy(actor)

# Perform soft parameter update
soft_update(actor, target, tau=0.005)
```

This code should work 'as is'.

### Hard Network Parameter Updates

???+ api "API Docs"

    [`velora.utils.hard_update(source, target)`](../reference/utils.md#velora.utils.torch.hard_update)

Are soft updates too slow? We can perform a hard parameter update (without the $\tau$ factor) using `hard_update` instead:

```python
from copy import deepcopy

from velora.models.sac.discrete import SACActorDiscrete
from velora.utils import hard_update

# Set two networks
actor = SACActorDiscrete(4, 10, 1)
target = deepcopy(actor)

# Perform hard parameter update
hard_update(actor, target)
```

This code should work 'as is'.

### Model Parameter Counts

???+ api "API Docs"

    [`velora.utils.total_parameters(model)`](../reference/utils.md#velora.utils.torch.total_parameters)

    [`velora.utils.active_parameters(model)`](../reference/utils.md#velora.utils.torch.active_parameters)

Ever been curious about the number of parameters a model has? We've got a few methods to quickly help with this!

`total_parameters()` for ALL parameters, and `active_parameters()` for ones in use:

```python
from velora.models import SACActorDiscrete
from velora.utils import total_parameters, active_parameters

model = SACActorDiscrete(4, 10, 1)

total_params = total_parameters(model)
active_params = active_parameters(model)
print(total_params, active_params)
```

This code should work 'as is'.

???+ warning "Active Parameters"

    `active_parameters` is useful when you first initialize a sparsely connected model. 
    
    Using it after training, returns the same result as `total_parameters` due to floating point precision errors. Sparsity masked weights are still close to `0` but with a small `+-` variance. 

---

That wraps up our utility methods and our user guide tutorials. Excellent work getting this far! 👏

Still eager to learn more? Try one of the options 👇:

<div class="grid cards" markdown>

-   :material-puzzle-edit:{ .lg .middle } __Customization__

    ---

    Learn how to create your own algorithms using Velora's building blocks.

    [:octicons-arrow-right-24: Read more](../customize/index.md)

-   :material-note-search:{ .lg .middle } __Theory__

    ---

    Read the theory behind the RL algorithms Velora uses.

    [:octicons-arrow-right-24: Start learning](../theory/index.md)

</div>
