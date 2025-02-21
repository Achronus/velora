# Working with Prebuilt Backbones

Backbones are a way of performing feature extraction techniques before passing through it through the core prediction algorithm.

Velora has two prebuilt options for this: an `MLP` and a `BasicCNN`.

## MLP

???+ api "API Docs"

    [`velora.models.backbone.MLP`](../reference/models/backbone.md#velora.models.backbone.MLP)

The `MLP` is a dynamic class for building Multi-layer Perceptron Networks - the traditional fully-connected neuron architecture.

Even though our algorithms focus on LNNs, we've added this into the framework on purpose to make it easy to compare the difference between the two for your own experiments.

The main component is the `n_hidden` parameter that is a `List[int]` (or a single `int` for one layer). This creates the `nn.Linear` hidden layers for you automatically.

It also comes with a few optional arguments, such as `activation` and `dropout_p`:

- `activation` - defines the activation function between the layers, default is `relu`.
- `dropout_p` - the dropout probability assigned between each layer, default is `0.0` meaning no dropout layers are applied.

Here's a code example:

```python
from velora.models.backbone import MLP
import torch

nn = MLP(
    in_features=4, 
    n_hidden=[256, 128, 64],  # (1) 
    out_features=2,
    activation="relu",  # (2)
    dropout_p=0.2  # (3)
)

x = torch.ones((1, 4))

y_pred = nn(x)
```

1. 3 hidden layers
2. Activation function used between the layers (optional)
3. Dropout used between layers, 20% probability (optional)

This code should work 'as is'.

## BasicCNN

???+ api "API Docs"

    [`velora.models.backbone.BasicCNN`](../reference/models/backbone.md#velora.models.backbone.BasicCNN)

The `BasicCNN` uses a static architecture from the DQN Nature paper: [Human-level control through deep reinforcement learning [:material-arrow-right-bottom:]](https://www.nature.com/articles/nature14236).

The paper used it for Atari games, but has been adopted in other libraries such as [Stable-Baselines3 [:material-arrow-right-bottom:]](https://stable-baselines3.readthedocs.io/en/master/index.html) as a go-to CNN architecture, so we thought we'd use the same one! 😊

As an added bonus, it makes things easier for comparing SB3 baselines with our algorithms 😉.

???+ note "Backbones with Velora algorithms"

    Currently, Velora doesn't directly use backbones in it's prebuilt algorithms, they are strictly LNN architectures. So, you need to manually apply them yourself (we'll show you how to do this shortly).
    
    We plan to change this in the future, but right now we are focusing on building a robust baseline for our algorithms.

To use the `BasicCNN` architecture, we pass in the number of `in_channels` and then can call the `forward()` or `out_size()` methods:

```python
from velora.models.backbone import BasicCNN
import torch

cnn = BasicCNN(1)

x = torch.ones((1, 64, 64))  # (in_channels, height, width)

n_feature_maps = cnn.out_size(x.size()[-2:])  # (1) 1024
y_pred = cnn(x.unsqueeze(0))  # (1, 1024)
```

1. Number of `in_features` to an NCP or MLP

This code should work 'as is'.

### Usage With a Custom LNN

To use the `BasicCNN` with a custom LNN, we can combine it with the `LiquidNCPNetwork` module:

```python
from velora.models import LiquidNCPNetwork
from velora.models.backbone import BasicCNN

import torch

cnn = BasicCNN(1)

x = torch.ones((1, 64, 64))  # (in_channels, height, width)

n_neurons = 10
out_features = 3

n_feature_maps = cnn.out_size(x.size()[-2:])  # (1) 1024

ncp = LiquidNCPNetwork(n_feature_maps, n_neurons, out_features)

cnn_pred = cnn(x.unsqueeze(0))  # (1, 1024)
y_pred, hidden = ncp(cnn_pred)
```

This code should work 'as is'.

Or, as a module:

```python
from typing import Tuple

from velora.models import LiquidNCPNetwork
from velora.models.backbone import BasicCNN

import torch
import torch.nn as nn


class LiquidCNN(nn.Module):
    """
    A basic Liquid Network with a CNN backbone.
    """
    def __init__(self, 
        img_shape: Tuple[int, int, int], 
        n_neurons: int, 
        out_features: int
    ) -> None:
        super().__init__()

        if len(img_shape) != 3:
            raise ValueError(
                f"Invalid '{img_shape=}'. Must be '(in_channels, height, width)'."
            )

        self.cnn = BasicCNN(img_shape[0])

        in_features = self.cnn.out_size(img_shape[-2:])
        self.ncp = LiquidNCPNetwork(in_features, n_neurons, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        out_features = self.cnn(x)
        y_pred, hidden = self.ncp(out_features)
        return y_pred, hidden


x = torch.ones((1, 64, 64))  # (in_channels, height, width)

model = LiquidCNN(x.shape, 10, 3)

y_pred, hidden = model(x.unsqueeze(0))
```

This code should work 'as is'.

---

And that completes the customization tutorials! :partying_face:

Well done for getting this far! 👏 And thanks for reading! 😍
