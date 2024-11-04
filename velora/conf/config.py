from pathlib import Path
from typing import Any
import yaml

from pydantic import BaseModel, ConfigDict, model_validator, validate_call
from pydantic_settings import BaseSettings

from velora.exc import IncorrectFileTypeError


class EnvironmentSettings(BaseModel):
    NAME: str
    EPISODES: int
    SEED: int | None = None


class EnvConfig(BaseSettings):
    ENV: EnvironmentSettings

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def validate_keys(cls, values: dict) -> dict:
        """Convert all dictionary keys to uppercase."""

        def convert_keys(data: dict[str, Any] | list) -> dict:
            if isinstance(data, dict):
                return {key.upper(): convert_keys(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [convert_keys(item) for item in data]

            return data

        return convert_keys(values)


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
def load_config(filepath: Path | str) -> EnvConfig:
    """Loads a YAML file as an EnvConfig Pydantic settings model."""
    yaml_config = load_yaml(filepath)
    return EnvConfig(**yaml_config)
