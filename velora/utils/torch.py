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
    Converts a list of items to a Tensor, then:

    1. Converts it to a specific `dtype`
    2. Loads it onto `device`

    Parameters:
        items (List[Any]): a list of items of any type
        dtype (torch.dtype, optional): the data type for the tensor
        device (torch.device, optional): the device to perform computations on
    """
    return torch.tensor(items).to(dtype).to(device)


def stack_tensor(
    items: List[torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Stacks a list of tensors together, then:

    1. Converts it to a specific `dtype`
    2. Loads it onto `device`

    Parameters:
        items (List[torch.Tensor]): a list of torch.Tensors full of items
        dtype (torch.dtype, optional): the data type for the tensor
        device (torch.device, optional): the device to perform computations on
    """
    return torch.stack(items).to(dtype).to(device)


def soft_update(source: nn.Module, target: nn.Module, tau: float = 0.005) -> None:
    """
    Performs a soft parameter update between two PyTorch Networks.

    Parameters:
        source (nn.Module): the source network
        target (nn.Module): the target network
        tau (float, optional): the soft update factor used to slowly update
            the target network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """
    Performs a hard parameter update between two PyTorch Networks.

    Parameters:
        source (nn.Module): the source network
        target (nn.Module): the target network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
