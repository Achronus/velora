from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Self, Tuple

import gymnasium as gym
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from velora.buffer.base import BufferBase  # pragma: no cover

from velora.models.config import ModuleConfig, RLAgentConfig, TrainConfig
from velora.models.lnn.ncp import LiquidNCPNetwork
from velora.utils.torch import summary


class NCPModule(nn.Module):
    """
    A base class for NCP modules.

    Useful for Actor-Critic modules.
    """

    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        device: torch.device | None = None,
    ):
        """
        Parameters:
            in_features (int): the number of input nodes
            n_neurons (int): the number of hidden neurons
            out_features (int): the number of output nodes
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__()

        self.device = device

        self.ncp = LiquidNCPNetwork(
            in_features=in_features,
            n_neurons=n_neurons,
            out_features=out_features,
            device=device,
        ).to(device)

    def config(self) -> ModuleConfig:
        """
        Gets details about the module.

        Returns:
            config (ModuleConfig): a config model containing module details.
        """
        return ModuleConfig(
            active_params=self.ncp.active_params,
            total_params=self.ncp.total_params,
            architecture=summary(self),
        )


class RLAgent:
    """
    A base class for RL agents.

    Provides a blueprint describing the core methods that agents *must* have and
    includes useful utility methods.
    """

    def __init__(self, device: torch.device | None) -> None:
        """
        Parameters:
            device (torch.device, optional): the device to perform computations on
        """
        self.device = device
        self.config: RLAgentConfig | None = None
        self.buffer: "BufferBase" | None = None

        self.active_params = 0
        self.total_params = 0

    @abstractmethod
    def train(
        self,
        env: gym.Env,
        batch_size: int,
        n_episodes: int,
        max_steps: int,
        window_size: int,
        *args,
        **kwargs,
    ) -> Any:
        pass  # pragma: no cover

    @abstractmethod
    def predict(
        self, state: torch.Tensor, hidden: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass  # pragma: no cover

    @abstractmethod
    def save(self, filepath: str | Path, *, buffer: bool = False) -> None:
        pass  # pragma: no cover

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path, *, buffer: bool = False) -> Self:
        pass  # pragma: no cover

    def _set_train_params(self, params: Dict[str, Any]) -> TrainConfig:
        """
        Helper method. Sets the `train_params` given a dictionary of training parameters.

        Parameters:
            params (Dict[str, Any]): a dictionary of training parameters

        Returns:
            config (TrainConfig): a training config model
        """
        return TrainConfig(
            callbacks=(
                dict(cb.config() for cb in params["callbacks"])
                if params["callbacks"]
                else None
            ),
            **{
                k: v for k, v in params.items() if k not in ["self", "env", "callbacks"]
            },
        )
