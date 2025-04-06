# Building You're Own Liquid Networks

Now that we know how to build our sparsely connected neurons, we can start building our own LNNs.

## Cells as Layers

???+ api "API Docs"

    [`velora.models.lnn.NCPLiquidCell`](../reference/models/lnn.md#velora.models.lnn.cell.NCPLiquidCell)

For this example, we'll focus on creating a network made purely of `NCPLiquidCells`.

Initializing the layers is the easy part:

```python
from collections import OrderedDict

from velora.models.lnn import NCPLiquidCell
from velora.wiring import Wiring

in_features = 4
n_neurons = 10
out_features = 1

wiring = Wiring(in_features, n_neurons, out_features)
masks, counts = wiring.data()

names = ["inter", "command", "motor"]
layers = [
    NCPLiquidCell(in_features, counts.inter, masks.inter),
    NCPLiquidCell(counts.inter, counts.command, masks.command),
    NCPLiquidCell(counts.command, counts.motor, masks.motor)
]
layers = OrderedDict([(name, layer) for name, layer in zip(names, layers)])
```

This code should work 'as is'.

However, using them is tricky. Since we are using a Recurrent architecture, we need to iterate through each layer manually and retain their respective hidden states.

This requires a bit of wizardry ✨. Let's look at some more code first:

```python hl_lines="5 10 17-18 21"
from typing import Tuple

import torch

out_sizes = [layer.n_hidden for layer in layers.values()]

def ncp_forward(x: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """"""
    h_state = torch.split(hidden, out_sizes, dim=1) # (1)

    new_h_state = []
    inputs = x

    # Handle layer independence
    for i, layer in enumerate(layers.values()):
        y_pred, h = layer(inputs, h_state[i]) # (2)
        inputs = y_pred  # (batch_size, layer_out_features)
        new_h_state.append(h)

    new_h_state = torch.cat(new_h_state, dim=1)  # (3) (batch_size, n_units)
    return y_pred, new_h_state
```

1. (1) Split hidden state -> layer batches
2. (2) Iterate over each layer -> store hidden state
3. (3) Merge the hidden states -> a single tensor

??? warning "layer(input) vs layer.forward(input)"

    In the previous code block we used `layer(input)` instead of `layer.forward(input)`. This is a best practice and recommended when using PyTorch `nn.Modules`. 

    PyTorch automatically invokes the layer's `__call__` method which performs the following steps:

    1. Checks if the module is in training or evaluation mode.
    2. Handles any registered hooks (pre-forward and forward hooks).
    3. Calls the `forward()` method.
    4. Handles any registered backward hooks.
    5. Takes care of autograd (automatic differentiation) mechanics.

    Using `layer.forward()` would only execute the forward pass logic, removing a lot of the added benefits such as:

    - Proper handling of training vs evaluation modes (affects dropout, batch norm, etc.).
    - Execution of any registered hooks that might be needed for debugging or monitoring.
    - Correct setup of the autograd graph for backpropagation.

    ❗ Therefore, it is highly recommended to use `layer(input)` rather than calling the `forward()` method directly to maximize the added benefits. 
    
    Keep an eye out for this when building your own models! 😉

Firstly, we calculate the `output sizes` for each layer and store them in a list.

Next, we go into the function, split our hidden state into batches that match the output sizes.

Then, we iterate over each layer, passing the layers prediction into the next layer, while storing the layer's hidden state in a list.

Lastly, we flatten the networks hidden states back into a single tensor and return the final layer's prediction with the flattened hidden state array.

Now we can start using our network! Given some input `x` and an empty hidden state, we can make a prediction:

```python
import torch

# Set our inputs
x = torch.tensor(
    [-6.6, -1.1, -5.98, -1.69]
).unsqueeze(0)

# Compute our hidden state size
n_units = n_neurons + out_features
# n_units = sum(out_sizes)  # Same as above!
batch_size = x.size()[0]

# Create a starting hidden state
h_state = torch.zeros((batch_size, n_units))

# Make a prediction
y_pred, h_state = ncp_forward(x, h_state)
```

### Full Code Example

Here's the complete code we've looked at:

```python
from collections import OrderedDict
from typing import Tuple

from velora.models.lnn.cell import NCPLiquidCell
from velora.wiring import Wiring

import torch

in_features = 4
n_neurons = 10
out_features = 1

wiring = Wiring(in_features, n_neurons, out_features)
masks, counts = wiring.data()

names = ["inter", "command", "motor"]
layers = [
    NCPLiquidCell(in_features, counts.inter, masks.inter),
    NCPLiquidCell(counts.inter, counts.command, masks.command),
    NCPLiquidCell(counts.command, counts.motor, masks.motor)
]
layers = OrderedDict([(name, layer) for name, layer in zip(names, layers)])

out_sizes = [layer.n_hidden for layer in layers.values()]

def ncp_forward(x: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """"""
    h_state = torch.split(hidden, out_sizes, dim=1)

    new_h_state = []
    inputs = x

    # Handle layer independence
    for i, layer in enumerate(layers.values()):
        y_pred, h = layer(inputs, h_state[i])
        inputs = y_pred  # (batch_size, layer_out_features)
        new_h_state.append(h)

    new_h_state = torch.cat(new_h_state, dim=1)  # (batch_size, n_units)
    return y_pred, new_h_state


# Set our inputs
x = torch.tensor(
    [-6.6, -1.1, -5.98, -1.69]
).unsqueeze(0)

# Compute our hidden state size
n_units = n_neurons + out_features
# n_units = sum(out_sizes)  # Same as above!
batch_size = x.size()[0]

# Create a starting hidden state
h_state = torch.zeros((batch_size, n_units))

# Make a prediction
y_pred, h_state = ncp_forward(x, h_state)
```

This code should work 'as is'.

And that's it! Now you can use cells individually! But honestly, all of that is a exhausting 🥱.

So instead, why don't we use a prebuilt version? 😉

## Prebuilt Networks

???+ api "API Docs"

    [`velora.models.lnn.LiquidNCPNetwork`](../reference/models/lnn.md#velora.models.lnn.ncp.LiquidNCPNetwork)

We can eliminate the need to do the previous step by using our prebuilt `LiquidNCPNetwork`.

Admittedly, it uses a different architecture but follows a relatively similar format.

??? abstract "The Real Architecture"

    For those interested, the real architecture consists of:

    - Two `SparseLinear` layers
    - One `NCPLiquidCell` layer

    We initially tested them as three `NCPLiquidCells` and found their stability lacking. We hypothesis that additional cells can interfere with the agent's ability to learn an accurate set of system dynamics.

    However, we still encourage you to test this out for yourselves! Exploration is not just for RL agents! 😉

The main thing here is that all you need to do is write two lines of code and boom 💥, you've got a working, and heavily tested Liquid Network! 😉

```python
from velora.models import LiquidNCPNetwork

x = torch.tensor([-6.6, -1.1, -5.98, -1.69]).unsqueeze(0)  # Must be 2d

ncp = LiquidNCPNetwork(4, 10, 1)
y_pred, h_state = ncp(x)
```

This code should work 'as is'.

Notice how we don't need to manually define a hidden state. It's done for us automatically! 😎

---

Next, we'll look at working with buffers! 🤩
