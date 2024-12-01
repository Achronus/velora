import pytest
import os
from pathlib import Path

import torch
import gymnasium as gym

from velora.config import Config, load_config


@pytest.fixture
def config_file() -> Path:
    return Path(os.path.dirname(__file__), "demo.yaml")


@pytest.fixture
def config(config_file: Path) -> Config:
    return load_config(config_file)


@pytest.fixture
def env(config: Config) -> gym.Env:
    return gym.make(config.env.name)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")
