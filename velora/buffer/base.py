from abc import abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Self, get_args

import torch

from velora.buffer.experience import BatchExperience, Experience
from velora.utils.torch import stack_tensor, to_tensor

StateDictKeys = Literal["buffer", "capacity", "device"]
BufferKeys = Literal["states", "actions", "rewards", "next_states", "dones"]


class BufferBase:
    """
    A base class for all buffers.
    """

    def __init__(self, capacity: int, *, device: torch.device | None = None) -> None:
        """
        Parameters:
            capacity (int): the total capacity of the buffer
            device (torch.device, optional): the device to perform computations on
        """
        self.capacity = capacity
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self.device = device

    def push(self, exp: Experience) -> None:
        """
        Stores an experience in the buffer.

        Parameters:
            exp (Experience): a single set of experience as an object
        """
        self.buffer.append(exp)

    def _batch(self, batch: List[Experience]) -> BatchExperience:
        """
        Helper method. Converts a `List[Experience]` into a `BatchExperience`.
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        return BatchExperience(
            states=stack_tensor(states, device=self.device),
            actions=stack_tensor(actions, device=self.device),
            rewards=to_tensor(rewards, device=self.device).unsqueeze(1),
            next_states=stack_tensor(next_states, device=self.device),
            dones=to_tensor(dones, device=self.device).unsqueeze(1),
        )

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
        return len(self.buffer)

    def state_dict(self) -> Dict[StateDictKeys, Any]:
        """
        Return a dictionary containing the buffers contents. Includes:

        - `buffer` - serialized arrays of `{states, actions, rewards, next_states, dones}`.
        - `capacity` - the maximum capacity of the buffer.
        - `device` - the device used for computations.

        Returns:
            state_dict (Dict[Literal["buffer", "capacity", "device"], Any]): a dictionary containing the current state of the buffer.
        """
        if len(self.buffer) > 0:
            states, actions, rewards, next_states, dones = zip(*self.buffer)

            serialized_buffer: Dict[BufferKeys, List[Any]] = {
                "states": stack_tensor(states).cpu().tolist(),
                "actions": stack_tensor(actions).cpu().tolist(),
                "rewards": to_tensor(rewards).cpu().tolist(),
                "next_states": stack_tensor(next_states).cpu().tolist(),
                "dones": to_tensor(dones).cpu().tolist(),
            }
        else:
            serialized_buffer = {key: [] for key in list(get_args(BufferKeys))}

        return {
            "buffer": serialized_buffer,
            "capacity": self.capacity,
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
            state_dict["capacity"],
            device=device,
        )

        data: Dict[BufferKeys, List[Any]] = state_dict["buffer"]

        # Recreate experiences from the parallel arrays
        for state, action, reward, next_state, done in zip(
            data["states"],
            data["actions"],
            data["rewards"],
            data["next_states"],
            data["dones"],
        ):
            buffer.push(
                Experience(
                    state=to_tensor(state, device=device),
                    action=to_tensor(action, device=device),
                    reward=reward,
                    next_state=to_tensor(next_state, device=device),
                    done=done,
                )
            )

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
