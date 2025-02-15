from typing import Any, List

import torch
import numpy as np


def set_seed(value: int) -> None:
    """Sets the random seed for the `PyTorch` and `NumPy` packages."""
    torch.manual_seed(value)
    np.random.seed(value)


def set_device(device: str = "auto") -> torch.device:
    """Sets the `PyTorch` device dynamically."""
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Device set to '{device}'.")
    return torch.device(device)


def to_tensor(
    item: List[Any],
    stack: bool = False,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    unsqueeze: int | None = None,
) -> torch.Tensor:
    """
    Converts an item to a Tensor. Includes some helpful utilities:
    1. Converting a tensor a specific `dtype`
    2. `unsqueezing` the tensor
    3. Loading it onto a `device`
    4. Stacking the item into a single tensor

    Parameters:
        item (List[Any]): a list of items of any type
        stack (bool, optional): whether to stack the values into a single
            tensor. Default is `False`
        dtype (torch.dtype, optional): the data type for the tensor.
            Default is `torch.float32`
        device (torch.device, optional): the PyTorch device to load tensors onto.
            Default is `None`
        unsqueeze (int, optional): the unsqueeze dimension for extending the
            tensor. Default is `None`
    """
    new_tensor = torch.stack(item) if stack else torch.tensor(item)

    if unsqueeze:
        new_tensor = new_tensor.unsqueeze(unsqueeze)

    return new_tensor.to(dtype).to(device)
