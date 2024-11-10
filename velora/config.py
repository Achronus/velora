from pathlib import Path
from typing import Any
import yaml

from pydantic import BaseModel, ConfigDict, validate_call
from pydantic_settings import BaseSettings

from velora.exc import IncorrectFileTypeError


class EnvironmentSettings(BaseModel):
    name: str
    episodes: int
    seed: int | None = None


class NetworkSettings(BaseModel):
    hidden_size: int = 256
    batch_size: int = 128


class OtherSettings(BaseModel):
    percentile: int | float | None = None
    solve_threshold: int | float | None = None


class AgentSettings(BaseModel):
    pass


class ControllerSettings(BaseModel):
    pass


class Config(BaseSettings):
    env: EnvironmentSettings
    optimizer: dict[str, Any] | None = None
    model: NetworkSettings = NetworkSettings()
    other: OtherSettings = OtherSettings()

    model_config = ConfigDict(extra="ignore")


@validate_call(validate_return=True)
def load_yaml(filepath: Path | str) -> dict:
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
