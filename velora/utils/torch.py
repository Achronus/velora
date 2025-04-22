from typing import Any, Dict, List

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

    Returns:
        tensor (torch.Tensor): the updated `torch.Tensor`.
    """
    return torch.tensor(items).to(dtype=dtype, device=device)


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

    Returns:
        tensor (torch.Tensor): the updated `torch.Tensor`.
    """
    return torch.stack(items).to(dtype=dtype, device=device)


def soft_update(source: nn.Module, target: nn.Module, *, tau: float = 0.005) -> None:
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


@torch.jit.ignore
def total_parameters(model: nn.Module) -> int:
    """
    Calculates the total number of parameters used in a PyTorch `nn.Module`.

    Parameters:
        model (nn.Module): a PyTorch module with parameters

    Returns:
        count (int): the total number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.jit.ignore
def active_parameters(model: nn.Module) -> int:
    """
    Calculates the active number of parameters used in a PyTorch `nn.Module`.
    Filters out parameters that are `0`.

    Parameters:
        model (nn.Module): a PyTorch module with parameters

    Returns:
        count (int): the total active number of parameters.
    """
    return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)


@torch.jit.ignore
def summary(module: nn.Module) -> Dict[str, str]:
    """
    Outputs a summary of a module and all it's sub-modules as a dictionary.

    Returns:
        summary (Dict[str, str]): key-value pairs for the network layout.
    """
    model_dict = {}

    for name, mod in module.named_children():
        if len(list(mod.children())) > 0:
            # If the module has submodules, recurse
            model_dict[name] = summary(mod)
        else:
            # If it's a leaf module, store its string representation
            model_dict[name] = str(mod)

    return model_dict
