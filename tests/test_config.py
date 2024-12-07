import pytest
from pathlib import Path
import yaml

from velora.exc import IncorrectFileTypeError
from velora.config import (
    AgentSettings,
    Config,
    EnvironmentSettings,
    ModelSettings,
    PolicySettings,
    TrainingSettings,
    load_config,
)


class TestLoadConfig:
    @pytest.fixture
    def invalid_file(self, tmp_path: Path) -> Path:
        config_file = tmp_path / "config.txt"
        with open(config_file, "w") as f:
            f.write("invalid config")
        return config_file

    @staticmethod
    def test_valid_config_loading(config_file: Path):
        config = load_config(config_file)

        checks = [
            isinstance(config, Config),
            isinstance(config.env, EnvironmentSettings),
            isinstance(config.model, ModelSettings),
            isinstance(config.training, TrainingSettings),
            isinstance(config.agent, AgentSettings),
            isinstance(config.policy, PolicySettings),
        ]
        assert all(checks)

    @staticmethod
    def test_invalid_file_type(invalid_file: Path):
        with pytest.raises(IncorrectFileTypeError):
            load_config(invalid_file)

    @staticmethod
    def test_extra_fields_ignored(tmp_path: Path):
        config_dict = {
            "env": {
                "name": "CartPole-v1",
            },
            "extra_section": {"should": "be ignored"},
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_file)
        assert not hasattr(config, "extra_section")

    @staticmethod
    def test_file_not_found():
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    @staticmethod
    def test_invalid_yaml_syntax(tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: syntax:")

        with pytest.raises(yaml.YAMLError):
            load_config(config_file)
