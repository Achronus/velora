from pathlib import Path

from velora.agent.base import AgentBase

import torch

from velora.policy.base import PyTorchModel


class OffPolicyAgent(AgentBase):
    """
    A base class for off-policy agents.

    Args:
        config_filepath (pathlib.Path | str): a YAML config filepath
        model (velora.policy.PyTorchModel): a model containing PyTorch model settings
        device (torch.device | str, optional): device to run computations on, such as `cpu`, `cuda`. When `auto` configures the device automatically (Default is `auto`)
        logging (bool, optional): a flag to disable analytic logging. If True creates a [Weights and Bias](https://wandb.ai/) instance (Default is `True`)
    """

    def __init__(
        self,
        config_filepath: Path | str,
        model: PyTorchModel,
        device: torch.device | str,
        logging: bool,
    ) -> None:
        super().__init__(config_filepath, model, device, logging)

        self.storage = ReplayBuffer()
