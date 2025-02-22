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

    [`velora.utils.set_seed(value)`](../reference/utils.md#velora.utils.core.set_seed)

Setting a seed controls the randomness for the `PyTorch` and `NumPy` packages, making your experiment results consistent and reproducible.

Simply, pass `set_seed` a numeric `value` and you are good to go!

```python
from velora.utils import set_seed

set_seed(64)
```

This code should work 'as is'.

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

## Torch

During your own experiments, you might find yourself in need of a quick way to convert data to a `PyTorch` tensor, or want to perform a parameter update between two networks.

We've got a few methods to help with this:

- `to_tensor` - converts a list of data to a `torch.Tensor`.
- `stack_tensor` - stacks a list of `torch.Tensors` into a single one.
- `soft_update` - performs a soft parameter update between two `torch.nn.Modules`.
- `hard_update` - performs a hard parameter update between two `torch.nn.Modules`.

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

Some algorithms, like [`DDPG`](../tutorial/ddpg.md), perform soft target parameter updates using a hyperparameter $\tau$.

This method performs that exact process. Given a `source` network, `target` network, and soft update factor `tau`, we iterate through our parameters and perform a soft update:

```python
from copy import deepcopy

from velora.models.ddpg import DDPGActor
from velora.utils import soft_update

# Set two networks
actor = DDPGActor(4, 10, 1)
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

from velora.models.ddpg import DDPGActor
from velora.utils import hard_update

# Set two networks
actor = DDPGActor(4, 10, 1)
target = deepcopy(actor)

# Perform hard parameter update
hard_update(actor, target)
```

This code should work 'as is'.

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
