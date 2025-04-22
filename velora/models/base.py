from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Self, Tuple, Type

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from velora.gym.wrap import add_core_env_wrappers
from velora.models.weight import WeightInitType
from velora.utils.core import set_seed
from velora.utils.torch import summary

if TYPE_CHECKING:
    from velora.buffer.base import BufferBase  # pragma: no cover
    from velora.models.nf.modules import (
        ActorModule,
        CriticModule,
        EntropyModule,
    )  # pragma: no cover

from velora.models.config import ModuleConfig, RLAgentConfig, TrainConfig
from velora.models.lnn.ncp import LiquidNCPNetwork, NCPNetwork

StateDictKeys = Literal["modules", "optimizers"]


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

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features
        self.device = device

        self.ncp = NCPNetwork(
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


class LiquidNCPModule(nn.Module):
    """
    A base class for Liquid NCP modules.

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

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features
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


class RLModuleAgent:
    """
    A base class for RL agents that use modules.

    Provides a blueprint describing the core methods that agents *must* have and
    includes useful utility methods.
    """

    def __init__(
        self,
        env: gym.Env,
        actor_neurons: int,
        critic_neurons: int,
        buffer_size: int,
        optim: Type[optim.Optimizer],
        device: torch.device | None,
        seed: int | None,
    ) -> None:
        """
        Parameters:
            env (gym.Env): Gymnasium environment to train on
            actor_neurons (int): number of decision nodes (inter and command nodes)
                for the actor
            critic_neurons (int): number of decision nodes (inter and command nodes)
                for the critic
            buffer_size (int): buffer capacity
            device (torch.device, optional): the device to perform computations on
            seed (int, optional): random number seed
        """
        self.env = env
        self.eval_env = add_core_env_wrappers(env, device)
        self.actor_neurons = actor_neurons
        self.critic_neurons = critic_neurons
        self.buffer_size = buffer_size
        self.optim = optim
        self.device = device
        self.seed = set_seed(seed)

        self.action_dim: int = (
            self.env.action_space.n.item()
            if isinstance(self.env.action_space, gym.spaces.Discrete)
            else self.env.action_space.shape[-1]
        )
        self.state_dim: int = self.env.observation_space.shape[-1]

        self.action_scale = None
        self.action_bias = None

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_scale = (
                torch.tensor(
                    self.env.action_space.high - self.env.action_space.low,
                    device=device,
                )
                / 2.0
            )
            self.action_bias = (
                torch.tensor(
                    self.env.action_space.high + self.env.action_space.low,
                    device=device,
                )
                / 2.0
            )

        self.config: RLAgentConfig | None = None
        self.buffer: "BufferBase" | None = None

        self.actor: "ActorModule" | None = None
        self.critic: "CriticModule" | None = None

        self.entropy: "EntropyModule" | None = None

        self.active_params = 0
        self.total_params = 0

        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def train(
        self,
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

    def state_dict(self) -> Dict[StateDictKeys, Dict[str, Any]]:
        """
        Retrieves the agent's module state dictionaries and splits them into
        categories.

        Returns:
            state_dict (Dict[Literal["modules", "optimizers"], Dict[str, Any]]): the agent's module state dicts categorized.
        """
        final_dict: Dict[StateDictKeys, Dict[str, Any]] = {
            "modules": {},
            "optimizers": {},
        }

        for module in [self.actor, self.critic, self.entropy]:
            if module is not None:
                state_dict: Dict[str, Any] = module.state_dict()

                for key, val in state_dict.items():
                    if "optim" in key:
                        final_dict["optimizers"][key] = val
                    else:
                        final_dict["modules"][key] = val

        return final_dict

    def set_metadata(self, values: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """
        Creates the agents metadata based on a given set of local variables.

        Parameters:
            values (Dict[str, Any]): local variables
            seed (int): randomly generated seed

        Returns:
            metadata (Dict[str, Any]): an updated dictionary of agent metadata.
        """
        metadata = {
            k: v for k, v in values.items() if k not in ["self", "__class__", "env"]
        }
        metadata["device"] = str(self.device) if self.device is not None else "cpu"
        metadata["optim"] = f"torch.optim.{self.optim.__name__}"
        metadata["seed"] = seed

        return metadata

    def _set_train_params(self, params: Dict[str, Any]) -> TrainConfig:
        """
        Helper method. Sets the `train_params` given a dictionary of training parameters.

        Parameters:
            params (Dict[str, Any]): a dictionary of training parameters

        Returns:
            config (TrainConfig): a training config model
        """
        params = dict(
            callbacks=(
                dict(cb.config() for cb in params["callbacks"])
                if params["callbacks"]
                else None
            ),
            **{k: v for k, v in params.items() if k not in ["self", "callbacks"]},
        )
        return TrainConfig(**params)
