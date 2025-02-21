from typing import List, Tuple

import torch
import torch.nn as nn

from velora.models.activation import ActivationEnum, ActivationTypeLiteral


class MLP(nn.Module):
    """
    A dynamic multi-layer perceptron architecture for feature extraction.

    !!! warning

        The network output (`y_pred`) is not passed through an activation function.

        This must be applied manually (if required).
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: List[int] | int,
        out_features: int,
        *,
        activation: ActivationTypeLiteral = "relu",
        dropout_p: float = 0.0,
    ) -> None:
        """
        Parameters:
            in_features (int): the number of input features
            n_hidden (List[int] | int): a `list` of hidden node sizes or
                a `single` hidden node size. Dynamically creates `nn.Linear`
                layers based on sizes
            out_features (int): the number of output features
            activation (str, optional): the type of activation function used
                between layers
            dropout_p (float, optional): the dropout probability rate used between
                layers
        """
        super().__init__()

        self.dropout_p = dropout_p
        n_hidden = [n_hidden] if isinstance(n_hidden, int) else n_hidden

        input = nn.Linear(in_features, n_hidden[0])
        h_layers = self._set_hidden_layers(n_hidden, activation)
        output = nn.Linear(n_hidden[-1], out_features)

        self.fc = nn.Sequential(
            input,
            ActivationEnum.get(activation),
            *h_layers,
            output,
        )

    def _set_hidden_layers(self, n_hidden: List[int], activation: str) -> nn.ModuleList:
        """
        Helper method. Dynamically creates the hidden layers with
        activation functions and dropout layers.

        Parameters:
            n_hidden (List[int]): a list of hidden node sizes
            activation (str): the name of the activation function

        Returns:
            mlp (nn.ModuleList): a list of `nn.Linear` layers.
        """
        h_layers = nn.ModuleList()

        for i in range(len(n_hidden) - 1):
            layers = [
                nn.Linear(n_hidden[i], n_hidden[i + 1]),
                ActivationEnum.get(activation),
            ]

            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(self.dropout_p))

            h_layers.extend(layers)

        return h_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Parameters:
            x (torch.Tensor): the input tensor

        Returns:
            y_pred (torch.Tensor): network predictions.
        """
        return self.fc(x)


class BasicCNN(nn.Module):
    """
    A CNN backbone from the DQN Nature paper: [Human-level control through deep reinforcement learning [:material-arrow-right-bottom:]](https://www.nature.com/articles/nature14236).

    Only contains the Convolutional Network with a flattened output.

    Useful for connecting with other architectures such as the
    `LiquidNCPNetwork`.
    """

    def __init__(self, in_channels: int) -> None:
        """
        Parameters:
            in_channels (int): the number of channels in the input image. E.g.,

                - `3` for RGB
                - `1` for grayscale
        """
        super().__init__()
        self.in_channels = in_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

    def out_size(self, dim: Tuple[int, int]) -> int:
        """
        Calculates the size of the convolution output using a dummy input.

        Often used as the `in_features` to linear layers.

        Parameters:
            dim (Tuple[int, int]): the `(height, width)` of a single image

        Returns:
            output_size (int): the number of feature maps.
        """
        if len(dim) != 2:
            raise ValueError(f"Invalid '{dim=}'. Should be '(height, width)'.")

        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, *dim)
            return self.conv(x).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): a batch of images in the shape
                `(batch_size, in_channels, height, width)`.

                - `batch_size` - number of images in the batch
                - `in_channels` - number of colour channels
                - `height` - height of the images
                - `width` - width of the images

        Returns:
            y_pred (torch.Tensor): the flattened predicted feature maps.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Invalid '{x.shape=}'. Should be `(batch_size, in_channels, height, width)`."
            )

        return self.conv(x)
