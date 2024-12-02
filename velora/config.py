from pathlib import Path
from typing import Any
import yaml

from pydantic import BaseModel, Field, validate_call
from pydantic_settings import BaseSettings, SettingsConfigDict

from velora.exc import IncorrectFileTypeError


class EnvironmentSettings(BaseModel):
    name: str


class ModelSettings(BaseModel):
    """PyTorch model settings."""

    hidden_size: int = 256
    batch_size: int = 128


class TrainingSettings(BaseModel):
    """Training loop settings."""

    episodes: int = 100
    timesteps: int = 1000
    seed: int | None = None


class AgentSettings(BaseModel):
    alpha: float = Field(default=0.01, gt=0)
    gamma: float = Field(default=0.9, ge=0, le=1)


class PolicySettings(BaseModel):
    epsilon: float = Field(default=1, ge=0, le=1)
    min_epsilon: float = Field(default=0.1, ge=0, le=1)
    decay_rate: float = Field(default=0.01, ge=0, le=1)


class ControllerSettings(BaseModel):
    pass


class Config(BaseSettings):
    env: EnvironmentSettings
    optimizer: dict[str, Any] | None = None
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    agent: AgentSettings = AgentSettings()
    policy: PolicySettings = PolicySettings()

    model_config = SettingsConfigDict(extra="ignore")


@validate_call(validate_return=True)
def load_yaml(filepath: Path | str) -> dict[str, Any]:
    """Loads a YAML file as a dictionary."""
    if not Path(filepath).exists():
        raise FileNotFoundError("File does not exist.")

    yaml_file = str(filepath).split(".")[-1] == "yaml"

    if not yaml_file:
        raise IncorrectFileTypeError(
            "Incorrect file type provided. Must be a 'yaml' file."
        )

    with open(filepath, "r") as f:
        yaml_config = yaml.safe_load(f)

    return yaml_config


@validate_call(validate_return=True)
def load_config(filepath: Path | str) -> Config:
    """Loads a YAML file as a Velora Config model."""
    yaml_config = load_yaml(filepath)
    return Config(**yaml_config)
