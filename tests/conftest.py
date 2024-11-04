import pytest
import os
from pathlib import Path


@pytest.fixture
def config_file() -> Path:
    return Path(os.path.dirname(__file__), "demo.yaml")
