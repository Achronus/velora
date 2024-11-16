import pytest
from pathlib import Path
import yaml

from velora.exc import IncorrectFileTypeError
from velora.config import Config, EnvironmentSettings, load_config


class TestGymEnvConfig:
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
            config.env.name == "CartPole-v1",
            config.env.episodes == 100,
            config.env.seed == 42,
        ]
        assert all(checks)

    @staticmethod
    def test_case_insensitive_loading(tmp_path: Path):
        """Test that keys are converted to uppercase regardless of input case."""
        config_dict = {
            "env": {
                "name": "CartPole-v1",
                "episodes": 100,
                "seed": 42,
            }
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_file)

        checks = [
            config.env.name == "CartPole-v1",
            config.env.episodes == 100,
            config.env.seed == 42,
        ]
        assert all(checks)

    @staticmethod
    def test_invalid_file_type(invalid_file: Path):
        with pytest.raises(IncorrectFileTypeError):
            load_config(invalid_file)

    @staticmethod
    def test_missing_required_fields(tmp_path: Path):
        config_dict = {
            "env": {
                "name": "CartPole-v1"
                # Missing required 'episodes' field
            }
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        with pytest.raises(ValueError):
            load_config(config_file)

    @staticmethod
    def test_optional_seed(tmp_path: Path):
        config_dict = {
            "env": {
                "name": "CartPole-v1",
                "episodes": 100,
                # No seed provided
            }
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_file)
        assert config.env.seed is None

    @staticmethod
    def test_extra_fields_ignored(tmp_path: Path):
        config_dict = {
            "env": {
                "name": "CartPole-v1",
                "episodes": 100,
                "seed": 42,
                "extra_field": "should be ignored",
            },
            "extra_section": {"should": "be ignored"},
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_file)

        checks = [
            not hasattr(config.env, "extra_field"),
            not hasattr(config, "extra_section"),
        ]
        assert all(checks)

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
