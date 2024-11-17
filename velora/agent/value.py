from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, PrivateAttr
import torch


class ValueFunction(ABC, BaseModel):
    """A base class for value functions."""

    _values: torch.Tensor = PrivateAttr(None)

    @property
    def shape(self) -> tuple[int, ...]:
        """Retrieves the shape of value function."""
        return self._values.shape

    @abstractmethod
    def model_post_init(self, __context: Any) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def update(self) -> None:
        """Update a value."""
        pass  # pragma: no cover

    @abstractmethod
    def __getitem__(self) -> float | torch.Tensor:
        pass  # pragma: no cover

    @abstractmethod
    def __repr__(self) -> str:
        pass  # pragma: no cover

    def __len__(self) -> int:
        """Computes the number of items in the function."""
        return self._values.shape.numel()


class V(ValueFunction):
    """
    A state-value function `V(s)`.

    Args:
        num_states (int): The number of possible states
        device (str, optional): The CPU or GPU device to load onto. Default is `cpu`
    """

    num_states: int
    device: str = "cpu"

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


class Q(ValueFunction):
    """
    An action-value function `Q(s, a)`.

    Args:
        num_states (int): The number of possible states
        num_actions (int): the number of possible actions
        device (str, optional): The CPU or GPU device to load onto. Default is `cpu`
    """

    num_states: int
    num_actions: int
    device: str = "cpu"

    def model_post_init(self, __context: Any) -> None:
        self._values = torch.zeros(
            (self.num_states, self.num_actions),
            device=self.device,
        )

    def update(self, state: int, action: int, value: float) -> None:
        """Update the action-value for a state-action pair."""
        self._values[state, action] = value

    def __getitem__(self, index: tuple[int, int] | int | slice) -> float | torch.Tensor:
        if isinstance(index, (slice, int)):
            return self._values[index]

        return self._values[index].item()

    def __repr__(self) -> str:
        return f"Q(s={self.num_states}, a={self.num_actions}, values={self._values})"
