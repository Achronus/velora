import torch
import torch.nn as nn
import torch.nn.functional as F

from velora.models.weight import WeightInitType, get_init_fn


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
        init_type: str | WeightInitType = "kaiming_uniform",
        bias: bool = True,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            in_features (int): number of input features
            out_features (int): number of output features
            mask (torch.Tensor): sparsity mask tensor of shape
                `(out_features, in_features)`
            init_type (str, optional): the type of weight initialization
            bias (bool, optional): a flag to enable additive bias
            device (torch.device, optional): device to perform computations on
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.register_buffer("mask", mask.to(device).detach())

        weight = torch.empty((out_features, in_features), device=device)
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(init_type)

        with torch.no_grad():
            self.weight.data.mul_(self.mask)

    def reset_parameters(self, style: str | WeightInitType) -> None:
        """
        Initializes weights and biases using an initialization method.
        """
        weight_fn = get_init_fn(style)
        weight_fn(self)

    def update_mask(self, mask: torch.Tensor) -> None:
        """
        Updates the sparsity mask with a new one.

        Parameters:
            mask (torch.Tensor): new mask
        """
        self.mask = mask.to(self.device).detach()

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
