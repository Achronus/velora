from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Self

import torch

from velora.buffer.experience import BatchExperience
from velora.utils.torch import to_tensor

StateDictKeys = Literal[
    "buffer",
    "capacity",
    "state_dim",
    "action_dim",
    "position",
    "size",
    "device",
]
BufferKeys = Literal["states", "actions", "rewards", "next_states", "dones"]


class BufferBase:
    """
    A base class for all buffers.

    Stores experiences `(states, actions, rewards, next_states, dones)` as
    individual items in tensors.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            capacity (int): the total capacity of the buffer
            state_dim (int): dimension of state observations
            action_dim (int): dimension of actions
            device (torch.device, optional): the device to perform computations on
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Position indicators
        self.position = 0
        self.size = 0

        # Pre-allocate storage
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Adds a single experience to the buffer.

        Parameters:
            state (torch.Tensor): current state observation
            action (torch.Tensor): action taken
            reward (float): reward received
            next_state (torch.Tensor): next state observation
            done (bool): whether the episode ended
        """
        self.states[self.position] = state.to(torch.float32).to(self.device)
        self.actions[self.position] = action.to(self.device)
        self.rewards[self.position] = to_tensor([reward], device=self.device)
        self.next_states[self.position] = next_state.to(torch.float32).to(self.device)
        self.dones[self.position] = to_tensor([done], device=self.device)

        # Update position - deque style
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @abstractmethod
    def sample(self) -> BatchExperience:
        """
        Samples experience from the buffer.

        Returns:
            batch (BatchExperience): an object of samples with the attributes (`states`, `actions`, `rewards`, `next_states`, `dones`).

                All items have the same shape `(batch_size, features)`.
        """
        pass  # pragma: no cover

    def __len__(self) -> int:
        """
        Gets the current size of the buffer.

        Returns:
            size (int): the current size of the buffer.
        """
        return self.size

    def state_dict(self) -> Dict[StateDictKeys, Any]:
        """
        Return a dictionary containing the buffers contents. Includes:

        - `buffer` - serialized arrays of `{states, actions, rewards, next_states, dones}`.
        - `capacity` - the maximum capacity of the buffer.
        - `state_dim` - state dimension.
        - `action_dim` - action dimension.
        - `position` - current buffer position.
        - `size` - current size of buffer.
        - `device` - the device used for computations.

        Returns:
            state_dict (Dict[str, Any]): a dictionary containing the current state of the buffer.
        """
        return {
            "buffer": {
                "states": self.states.cpu(),
                "actions": self.actions.cpu(),
                "rewards": self.rewards.cpu(),
                "next_states": self.next_states.cpu(),
                "dones": self.dones.cpu(),
            },
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "device": str(self.device) if self.device else None,
        }

    def save(self, filepath: str | Path) -> None:
        """
        Saves a buffers state to a file.

        Parameters:
            filepath (str | Path): where to save the buffer state

        Example Usage:
        ```python
        from velora.buffer import ReplayBuffer

        buffer = ReplayBuffer(capacity=100, device="cpu")

        buffer.save('checkpoints/buffer_100_cpu.pt')
        ```
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.state_dict()
        torch.save(state_dict, save_path)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """
        Restores the buffer from a saved state.

        Parameters:
            filepath (str | Path): buffer state file location

        Example Usage:
        ```python
        from velora.buffer import ReplayBuffer

        buffer = ReplayBuffer.load('checkpoints/buffer_100_cpu.pt')
        ```

        Returns:
            buffer (Self): a new buffer filled instance.
        """
        state_dict: Dict[StateDictKeys, Any] = torch.load(filepath)
        device = (
            torch.device(state_dict["device"])
            if state_dict["device"] is not None
            else None
        )

        buffer = cls(
            capacity=state_dict["capacity"],
            state_dim=state_dict["state_dim"],
            action_dim=state_dict["action_dim"],
            device=device,
        )

        data: Dict[BufferKeys, torch.Tensor] = state_dict["buffer"]

        buffer.position = state_dict["position"]
        buffer.size = state_dict["size"]
        buffer.states = data["states"].to(buffer.device)
        buffer.actions = data["actions"].to(buffer.device)
        buffer.rewards = data["rewards"].to(buffer.device)
        buffer.next_states = data["next_states"].to(buffer.device)
        buffer.dones = data["dones"].to(buffer.device)

        return buffer

    @staticmethod
    def create_filepath(filepath: str | Path) -> Path:
        """
        Updates a given `filepath` and converts it into a `buffer` friendly one.

        Parameters:
            filepath (str | Path): a filepath to convert

        Returns:
            buffer_path (Path): a buffer friendly filepath in the form `<filepath>.buffer.<filepath_ext>`.
        """
        path = Path(filepath)
        extension = path.name.split(".")[-1]
        buffer_name = path.name.replace(extension, f"buffer.{extension}")
        return path.with_name(buffer_name)
