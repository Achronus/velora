from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
import torch

from velora.utils.validation import device_validation


class ValueFunction(ABC, BaseModel):
    """A base class for value functions."""

    _values: torch.Tensor = PrivateAttr(None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def values(self) -> torch.Tensor:
        """Returns the values."""
        return self._values

    @property
    def shape(self) -> tuple[int, ...]:
        """Retrieves the shape of the values."""
        return self._values.shape

    @abstractmethod
    def model_post_init(self, __context: Any) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update a value."""
        pass  # pragma: no cover

    @abstractmethod
    def __getitem__(self, index: Any) -> float | torch.Tensor:
        pass  # pragma: no cover

    @abstractmethod
    def __repr__(self) -> str:
        pass  # pragma: no cover

    def as_state_values(self) -> list[float]:
        """Returns the state-action pairs as a 1D list of state-values. Suitable for reshaping to visualize."""
        return [torch.max(value).item() for value in self._values]

    def __len__(self) -> int:
        """Returns the total number of values."""
        return self._values.shape.numel()


class VTable(ValueFunction):
    """
    A state-value function `V(s)`. Stores all values as a torch.Tensor, acting like a lookup table.

    Args:
        num_states (int): The number of possible states
        device (str | torch.device, optional): The CPU or GPU device to load onto. Default is `cpu`
    """

    num_states: int
    device: str | torch.device = Field("cpu", validate_default=True)

    @field_validator("device")
    def validate_device(cls, device: str | torch.device) -> torch.device:
        return device_validation(device)

    def model_post_init(self, __context: Any) -> None:
        self._values = torch.zeros(
            self.num_states,
            device=self.device,
        )

    def update(self, state: int, value: float) -> None:
        """Update the state-value for a state."""
        self._values[state] = value

    def __getitem__(self, index: int | slice) -> float | torch.Tensor:
        if isinstance(index, slice):
            return self._values[index]

        return self._values[index].item()

    def __repr__(self) -> str:
        return f"V(s={self.num_states}, values={self._values})"


class QTable(ValueFunction):
    """
    An action-value function `Q(s, a)`. Stores all values as a torch.Tensor, acting like a lookup table.

    Args:
        num_states (int): The number of possible states
        num_actions (int): the number of possible actions
        device (str | torch.device, optional): The CPU or GPU device to load onto. Default is `cpu`
    """

    num_states: int
    num_actions: int
    device: str | torch.device = Field("cpu", validate_default=True)

    @field_validator("device")
    def validate_device(cls, device: str | torch.device) -> torch.device:
        return device_validation(device)

    def model_post_init(self, __context: Any) -> None:
        self._values = torch.zeros(
            (self.num_states, self.num_actions),
            device=self.device,
        )

    def update(self, state: int, action: int, value: float) -> None:
        """Update the action-value for a state-action pair."""
        self._values[state, action] = value

    def as_state_values(self) -> list[float]:
        """Returns the state-action pairs as a 1D list of state-values. Suitable for reshaping to visualize."""
        return [torch.max(value).item() for value in self._values]

    def __getitem__(self, index: tuple[int, int] | int | slice) -> float | torch.Tensor:
        if isinstance(index, (slice, int)):
            return self._values[index]

        return self._values[index].item()

    def __repr__(self) -> str:
        return f"Q(s={self.num_states}, a={self.num_actions}, values={self._values})"
