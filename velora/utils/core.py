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
