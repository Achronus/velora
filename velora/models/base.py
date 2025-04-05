from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Self, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from velora.models.weight import WeightInitType
from velora.utils.torch import summary

if TYPE_CHECKING:
    from velora.buffer.base import BufferBase  # pragma: no cover

from velora.models.config import (
    EpisodeTrainConfig,
    ModuleConfig,
    RLAgentConfig,
    RolloutTrainConfig,
)
from velora.models.lnn.ncp import LiquidNCPNetwork


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
        init_type: str | WeightInitType = "kaiming_uniform",
        device: torch.device | None = None,
    ):
        """
        Parameters:
            in_features (int): the number of input nodes
            n_neurons (int): the number of hidden neurons
            out_features (int): the number of output nodes
            init_type (str, optional): the type of weight initialization
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__()

        self.device = device

        self.ncp = LiquidNCPNetwork(
            in_features=in_features,
            n_neurons=n_neurons,
            out_features=out_features,
            init_type=init_type,
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

    def __init__(
        self,
        state_dim: int,
        n_neurons: int,
        action_dim: int,
        buffer_size: int,
        device: torch.device | None,
    ) -> None:
        """
        Parameters:
            state_dim (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes).
            action_dim (int): number of outputs (motor nodes)
            buffer_size (int): buffer capacity
            device (torch.device, optional): the device to perform computations on
        """
        self.state_dim = state_dim
        self.n_neurons = n_neurons
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device

        self.config: RLAgentConfig | None = None
        self.buffer: "BufferBase" | None = None

        self.actor: NCPModule | None = None
        self.critic: NCPModule | None = None

        self.actor_target: NCPModule | None = None
        self.critic_target: NCPModule | None = None

        self.actor_optim: optim.Optimizer | None = None
        self.critic_optim: optim.Optimizer | None = None

        self.active_params = 0
        self.total_params = 0

    @abstractmethod
    def train(
        self,
        env: gym.Env,
        n_episodes: int,
        max_steps: int,
        window_size: int,
        *args,
        **kwargs,
    ) -> Any:
        pass  # pragma: no cover

    @abstractmethod
    def predict(
        self,
        state: torch.Tensor,
        hidden: torch.Tensor,
        train_mode: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass  # pragma: no cover

    @abstractmethod
    def save(
        self,
        dirpath: str | Path,
        *,
        buffer: bool = False,
        config: bool = False,
    ) -> None:
        """
        Saves the current model state into `safetensors` and `json` files.

        !!! warning

            `model_config.json` is stored in the `dirpath.parent`.

        Includes:

        - `model_config.json` - contains the core details of the agent (optional)
        - `metadata.json` - contains the model, optimizer and buffer (optional) metadata
        - `model_state.safetensors` - contains the model weights and biases
        - `optim_state.safetensors` - contains the optimizer states (actor and critic)
        - `buffer_state.safetensors` - contains the buffer state (only if `buffer=True`)

        Parameters:
            dirpath (str | Path): the location to store the model state. Should only
                consist of `folder` names. E.g., `<folder>/<folder>`
            buffer (bool, optional): a flag for storing the buffer state
            config (bool, optional): a flag for storing the model's config
        """
        pass  # pragma: no cover

    @classmethod
    @abstractmethod
    def load(cls, dirpath: str | Path, *, buffer: bool = False) -> Self:
        """
        Creates a new agent instance by loading a saved one from the `dirpath`.
        Also, loads the original training buffer if `buffer=True`.

        These files must exist in the `dirpath`:

        - `metadata.json` - contains the model, optimizer and buffer (optional) metadata
        - `model_state.safetensors` - contains the model weights and biases
        - `optim_state.safetensors` - contains the optimizer states (actor and critic)
        - `buffer_state.safetensors` - contains the buffer state (only if `buffer=True`)

        Parameters:
            dirpath (str | Path): the location to store the model state. Should only
                consist of `folder` names. E.g., `<folder>/<folder>`
            buffer (bool, optional): a flag for storing the buffer state

        Returns:
            agent (Self): a new agent instance with the saved state
        """
        pass  # pragma: no cover

    def _set_train_params(
        self,
        params: Dict[str, Any],
        type: Literal["episode", "rollout"] = "episode",
    ) -> EpisodeTrainConfig | RolloutTrainConfig:
        """
        Helper method. Sets the `train_params` given a dictionary of training parameters.

        Parameters:
            params (Dict[str, Any]): a dictionary of training parameters
            type (Literal["episode", "rollout"], optional): the type of train
                config

        Returns:
            config (TrainConfig): a training config model
        """
        params = dict(
            callbacks=(
                dict(cb.config() for cb in params["callbacks"])
                if params["callbacks"]
                else None
            ),
            **{
                k: v for k, v in params.items() if k not in ["self", "env", "callbacks"]
            },
        )

        if type == "episode":
            return EpisodeTrainConfig(**params)

        return RolloutTrainConfig(**params)
