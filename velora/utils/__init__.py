from typing import Any

from pydantic import ConfigDict, validate_call
import torch

from velora.config import load_yaml, load_config
from velora.utils.plots import plot_state_values


__all__ = [
    "load_yaml",
    "load_config",
    "plot_state_values",
]


@validate_call(validate_return=True)
def ignore_empty_dicts(values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Creates a new dictionary with empty sub dictionaries removed."""
    return {k: v for k, v in values.items() if v != {}}


@validate_call(config=ConfigDict(arbitrary_types_allowed=True), validate_return=True)
def set_device() -> torch.device:
    """Automatically sets the PyTorch device to CUDA if available. CPU, otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
