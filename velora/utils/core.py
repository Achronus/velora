import random

import numpy as np
import torch


def set_seed(value: int | None = None) -> int:
    """
    Sets the random seed for `Python`, `PyTorch` and `NumPy`.
    When `None` will create a new one automatically.

    Parameters:
        value (int, optional): the seed value

    Returns:
        seed (int): the used seed value
    """
    if value is None:
        value = random.randint(0, 2**32 - 1)

    random.seed(value)
    torch.manual_seed(value)
    np.random.seed(value)

    return value


def set_device(device: str = "auto") -> torch.device:
    """
    Sets the `PyTorch` device dynamically.

    Parameters:
        device (str, optional): the name of the device to perform computations on.

            When `auto`:

            - Set to `cuda:0`, if available.
            - Else, `cpu`.

    Returns:
        device (torch.device): the `PyTorch` device.
    """
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return torch.device(device)
