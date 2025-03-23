import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLinear(nn.Module):
    """A `torch.nn.Linear` layer with sparsely weighted connections."""

    bias: torch.Tensor
    mask: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        *,
        bias: bool = True,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            in_features (int): number of input features
            out_features (int): number of output features
            mask (torch.Tensor): sparsity mask tensor of shape
                `(out_features, in_features)`
            bias (bool, optional): a flag to enable additive bias
            device (torch.device, optional): device to perform computations on
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("mask", mask.to(device).detach())

        weight = torch.empty((out_features, in_features), device=device)
        # self.weight = SparseParameter(weight, self.mask)
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        with torch.no_grad():
            self.weight.data.mul_(self.mask)

    def reset_parameters(self) -> None:
        """
        Initializes weights and biases using Kaiming uniform initialization.

        Same operation as `torch.nn.Linear`.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the layer.

        Parameters:
            x (torch.Tensor): input tensor with shape `(..., in_features)`

        Returns:
            y_pred (torch.Tensor): layer prediction with sparsity applied with shape `(..., out_features)`.
        """
        return F.linear(x, self.weight * self.mask, self.bias)

    def extra_repr(self) -> str:
        """String representation of layer parameters."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
