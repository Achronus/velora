from pathlib import Path
from typing import Any
import yaml

from pydantic import BaseModel, ConfigDict, Field, validate_call
from pydantic_settings import BaseSettings, SettingsConfigDict

from velora.exc import IncorrectFileTypeError


class EnvironmentSettings(BaseModel):
    """
    [Gymnasium](https://gymnasium.farama.org/) environment settings.

    Args:
        name (str): the name of environment
        n_envs (int, optional): the number of parallel environments (Default is `3`)
        gamma (float, optional): the discount factor used in the exponential moving average in the [NormalizeReward](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.NormalizeReward) wrapper (Default is `0.99`)
        epsilon (float, optional): stability parameter for normalization scaling. Used in the [NormalizeObservation](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.NormalizeObservation) and [NormalizeReward](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.NormalizeReward) wrappers (Default is `1e-8`)
        max_obs (float, optional): the max absolute value for observations. Used in the [RescaleObservation](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.RescaleObservation) wrapper (Default is `10.0`)
        max_reward (float, optional): the max absolute value for discounted return. Used in the [ClipReward](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.ClipReward) wrapper (Default is `10.0`)
    """

    name: str
    n_envs: int = 3
    gamma: float = 0.99
    epsilon: float = 1e-8
    max_obs: float = 10.0
    max_reward: float = 10.0


class RunSettings(BaseModel):
    """
    Training loop settings.

    Args:
        timesteps (int, optional): total number of timesteps the agent can iterate through in the environment (Default is `10_000`)
        log_count (int, optional): the number of iterations to
    """

    timesteps: int = 10000
    log_count: int = 100
    seed: int = 23


class OptimizerSettings(BaseModel):
    lr: float = 0.001

    model_config = ConfigDict(extra="allow")


class AgentSettings(BaseModel):
    alpha: float = Field(default=0.01, gt=0)
    gamma: float = Field(default=0.9, ge=0, le=1)


class PolicySettings(BaseModel):
    epsilon: float = Field(default=1, ge=0, le=1)
    min_epsilon: float = Field(default=0.1, ge=0, le=1)
    decay_rate: float = Field(default=0.01, ge=0, le=1)


class Config(BaseSettings):
    env: EnvironmentSettings
    optimizer: OptimizerSettings = OptimizerSettings()
    model: ModelSettings = ModelSettings()
    run: RunSettings = RunSettings()
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
