from typing import Any, List

import torch
import torch.nn as nn


def to_tensor(
    items: List[Any],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Converts a list of items to a Tensor. Includes some helpful utilities:
    1. Converting a tensor a specific `dtype`
    3. Loading it onto a `device`

    Parameters:
        items (List[Any]): a list of items of any type
        dtype (torch.dtype, optional): the data type for the tensor.
            Default is `torch.float32`
        device (torch.device, optional): the PyTorch device to load tensors onto.
            Default is `None`
    """
    return torch.tensor(items).to(dtype).to(device)


def stack_tensor(
    items: List[torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Stacks a list of tensors together. Includes some helpful utilities:
    1. Converting the tensor to a specific `dtype`
    3. Loading it onto a `device`

    Parameters:
        items (List[torch.Tensor]): a list of torch.Tensors full of items
        dtype (torch.dtype, optional): the data type for the tensor.
            Default is `torch.float32`
        device (torch.device, optional): the PyTorch device to load tensors onto.
            Default is `None`
    """
    return torch.stack(items).to(dtype).to(device)


def soft_update(source: nn.Module, target: nn.Module, tau: float = 0.005) -> None:
    """Performs a soft parameter update between two PyTorch Networks."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Performs a hard parameter update between two PyTorch Networks."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
