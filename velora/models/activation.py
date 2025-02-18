from enum import Enum
from typing import Literal, Self

import torch
import torch.nn as nn


class LeCunTanh(nn.Module):
    """
    Implements LeCun's Tanh activation function.

    Constants are applied to keep the variance of the output close to 1.
    """

    def __init__(self) -> None:
        super().__init__()

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * self.tanh(0.666 * x)


ActivationTypeLiteral = Literal[
    "relu",
    "tanh",
    "elu",
    "leaky_relu",
    "prelu",
    "selu",
    "silu",
    "softsign",
    "sigmoid",
    "hardsigmoid",
    "lecun_tanh",
]


class ActivationEnum(Enum):
    """An Enum for PyTorch activation functions."""

    RELU = nn.ReLU()
    TANH = nn.Tanh()
    ELU = nn.ELU()
    LEAKY_RELU = nn.LeakyReLU()
    PRELU = nn.PReLU()
    SELU = nn.SELU()
    SILU = nn.GELU()
    SOFTSIGN = nn.Softsign()
    SIGMOID = nn.Sigmoid()
    HARDSIGMOID = nn.Hardsigmoid()
    LECUN_TANH = LeCunTanh()

    @classmethod
    def get(cls, name: str | Self) -> nn.Module:
        """Get the `torch.nn` activation function."""
        if isinstance(name, cls):
            return name.value
        try:
            return cls[name.upper()].value
        except KeyError:
            raise ValueError(f"Unsupported activation function: {name}")


__all__ = [
    "LeCunTanh",
    "ActivationTypeLiteral",
    "ActivationEnum",
]
