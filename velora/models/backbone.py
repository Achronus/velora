from typing import List

import torch
import torch.nn as nn

from velora.models.activation import ActivationEnum, ActivationTypeLiteral


class MLP(nn.Module):
    """
    A dynamic multi-layer perceptron architecture for feature extraction.

    Parameters:
        in_features (int): the number of input features
        n_hidden (List[int] | int): a list of hidden node sizes or
            a single hidden node size. Dynamically creates Linear layers
            based on sizes
        out_features (int): the number of output features
        activation (ActivationTypeLiteral, optional): the type of activation function to use between layers. Default is 'relu'
        dropout_p (float, optional): a dropout probability rate. Default is '0.0'
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
        super().__init__()

        self.dropout_p = dropout_p
        n_hidden = [n_hidden] if isinstance(n_hidden, int) else n_hidden

        input = nn.Linear(in_features, n_hidden[0])
        h_layers = self.set_hidden_layers(n_hidden, activation)
        output = nn.Linear(n_hidden[-1], out_features)

        self.fc = nn.Sequential(
            input,
            ActivationEnum.get(activation),
            *h_layers,
            output,
        )

    def set_hidden_layers(self, n_hidden: List[int], activation: str) -> nn.ModuleList:
        """
        Dynamically creates the hidden layers with
        activation functions and dropout layers.
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
        Perform a forward pass through the network
        without an output activation.

        Parameters:
            x (torch.Tensor): the input tensor
        """
        return self.fc(x)
