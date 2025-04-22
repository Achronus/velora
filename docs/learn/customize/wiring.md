# Creating Sparse Neurons

A major part of the NCP algorithm is *sparse connections*. This is handled through the `Wiring` class which is then used by our `SparseLinear` module.

## Wiring and Layer Masks

???+ api "API Docs"

    [`velora.wiring.Wiring(in_features, n_neurons, out_features)`](../../learn/reference/wiring.md#velora.wiring.Wiring)

The `Wiring` class's main purpose is to create sparsity masks for the three NCP layers and store the information in dataclasses: `NeuronCounts`, `SynapseCounts`, and `LayerMasks`.

### Basic Usage

To use it, we create an instance of the `Wiring` class and then call the `data()` method to retrieve the `NeuronCounts` and `LayerMasks`:

???+ note "SynapseCounts"

    `SynapseCounts` is strictly used internally inside the wiring class. Typically, you won't need to access this or apply it elsewhere. 
    
    However, you can access it using the `n_connections` attribute if you need to.

```python
from velora.wiring import Wiring

wiring = Wiring(in_features=3, n_neurons=10, out_features=1)
masks, counts = wiring.data()
```

This code should work 'as is'.

The rest is all done for you automatically behind the scenes.

### Creating an NCP Network

Then, to create your own NCP network, you use them like this:

```python
from velora.models.lnn import NCPLiquidCell
from velora.wiring import Wiring

in_features = 3

wiring = Wiring(in_features, n_neurons=10, out_features=1)
masks, counts = wiring.data()

layers = [
    NCPLiquidCell(in_features, counts.inter, masks.inter),
    NCPLiquidCell(counts.inter, counts.command, masks.command),
    NCPLiquidCell(counts.command, counts.motor, masks.motor)
]
```

This code should work 'as is'.

There is also an optional `sparsity_level` parameter that controls the connection sparsity between neurons:

1. When `0.1` neurons are very dense, close to a traditional Neural Network.
2. When `0.9` neurons are extremely sparse.

```python hl_lines="5"
wiring = Wiring(
    in_features, 
    n_neurons=10, 
    out_features=1,
    sparsity_level=0.5 # (1)
)
```

1. Optional

Experimenting with this could be interesting for your own use cases.

We've found `0.5` to be optimal (which is default) for most cases, providing a decent balance between training speed and performance. So, we recommend you start with this first! 😊

## Dataclasses

When calling `wiring.data()` we receive two dataclasses: `NeuronCounts` and `LayerMasks`.

Both are designed to be simple and intuitive.

### NeuronCounts

???+ api "API Docs"

    [`velora.wiring.NeuronCounts`](../../learn/reference/wiring.md#velora.wiring.NeuronCounts)

`NeuronCounts` holds the counts for each type of node:

```python
from dataclasses import dataclass

@dataclass
class NeuronCounts:
    """
    Storage container for NCP neuron category counts.

    Parameters:
        sensory (int): number of input nodes
        inter (int): number of decision nodes
        command (int): number of high-level decision nodes
        motor (int): number of output nodes
    """

    sensory: int
    inter: int
    command: int
    motor: int
```

### LayerMasks

???+ api "API Docs"

    [`velora.wiring.LayerMasks`](../../learn/reference/wiring.md#velora.wiring.LayerMasks)

`LayerMasks` holds the created sparsity masks for each NCP layer:

```python
from dataclasses import dataclass

import torch

@dataclass
class LayerMasks:
    """
    Storage container for layer masks.

    Parameters:
        inter (torch.Tensor): sparse weight mask for input layer
        command (torch.Tensor): sparse weight mask for hidden layer
        motor (torch.Tensor): sparse weight mask for output layer
        recurrent (torch.Tensor): sparse weight mask for recurrent connections
    """

    inter: torch.Tensor
    command: torch.Tensor
    motor: torch.Tensor
    recurrent: torch.Tensor

```

The masks will vary depending on the network size and the `seed` you set. They are random connections after all!

???+ tip "Want to know how they work?"

    You can read more about them in the [Theory - Liquid Neural Networks](../theory/lnn.md) page.

We highly recommend you set a `seed` using the `set_seed` utility method first before creating a `Wiring` instance. This will help you maintain reproducibility between experiments:

```python
from velora.utils import set_seed
from velora.wiring import Wiring

set_seed(64)

wiring = Wiring(in_features=3, n_neurons=10, out_features=1)
masks, counts = wiring.data()
```

This code should work 'as is'.

### SynapseCounts

???+ api "API Docs"

    [`velora.wiring.SynapseCounts`](../../learn/reference/wiring.md#velora.wiring.SynapseCounts)

`SynapseCounts` holds the synapse connection counts for each node type:

```python
@dataclass
class SynapseCounts:
    """
    Storage container for NCP neuron synapse connection counts.

    Parameters:
        sensory (int): number of connections for input nodes
        inter (int): number of connections for decision nodes
        command (int): number of connections for high-level decision nodes
        motor (int): number of connections for output nodes
    """

    sensory: int
    inter: int
    command: int
    motor: int
```

As we've discussed, you likely won't ever need to use this. It's strictly used internally inside the `Wiring` class.

To access it through a created `Wiring` class instance, we use the `n_connections` attribute:

```python
from velora.utils import set_seed
from velora.wiring import Wiring

set_seed(64)

wiring = Wiring(in_features=3, n_neurons=10, out_features=1)
wiring.n_connections

# SynapseCounts(sensory=3, inter=2, command=4, motor=2)
```

This code should work 'as is'.

### Importing Dataclasses

If you need to work with the dataclasses directly, you can manually import them from the `wiring` module:

```python
from velora.wiring import LayerMasks, NeuronCounts, SynapseCounts

n_sensory = 1

counts = NeuronCounts(sensory=n_sensory, inter=6, command=4, motor=2)

masks = LayerMasks(
    inter=torch.zeros(
        (n_sensory, counts.inter),
        dtype=torch.int32,
    ),
    command=torch.zeros(
        (counts.inter, counts.command),
        dtype=torch.int32,
    ),
    motor=torch.zeros(
        (counts.command, counts.motor),
        dtype=torch.int32,
    ),
)

connections = SynapseCounts(sensory=1, inter=2, command=3, motor=2)
```

This code should work 'as is'.

## Sparse Linear Layers

???+ api "API Docs"

    [`velora.models.lnn.SparseLinear(in_features, out_features, mask)`](../../learn/reference/models/lnn.md#velora.models.lnn.sparse.SparseLinear)

We've seen how to use the `Wiring` class in `NCPLiquidCells` but what about in `Linear` layers?

We've created our own implementation for this called a `SparseLinear` layer that applies the sparsity mask to the weights automatically.

You can implement one like this:

```python
from velora.models.lnn import SparseLinear
from velora.wiring import Wiring

import torch

wiring = Wiring(4, 10, 1)
l1 = SparseLinear(4, 1, torch.abs(wiring.masks.motor.T))
```

This code should work 'as is'.

Notice how we transpose (`T`) the mask and then take it's absolute value.

The transpose operation is required to ensure our mask fits the weights correctly and take the absolute to ensure gradient stability (turning `-1` -> `1`).

We didn't have to do this in the `NCPLiquidCell` because they have their own separate mask processing! 😉

---

Up next we will look at the methods available for building Liquid Networks. See you there! 👋
