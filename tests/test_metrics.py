import os
from unittest.mock import patch, MagicMock

from velora.metrics.db import get_db_engine


class TestGetDBEngine:
    def test_active_test_mode(self):
        os.environ["VELORA_TEST_MODE"] = "True"
        with patch("velora.metrics.db.create_engine") as mock_create_engine:
            # Setup the mock engine
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            # Call the function
            engine = get_db_engine()

            # Assert create_engine was called with the right URL
            mock_create_engine.assert_called_once_with("sqlite:///:memory:")

            # Assert that SQLModel.metadata.create_all was called with our engine
            assert mock_engine == engine

    def test_default_mode(self):
        os.environ["VELORA_TEST_MODE"] = "False"
        with patch("velora.metrics.db.create_engine") as mock_create_engine:
            # Setup the mock engine
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            # Call the function
            engine = get_db_engine()

            # Assert create_engine was called with the right URL
            mock_create_engine.assert_called_once_with("sqlite:///metrics.db")

            # Assert that SQLModel.metadata.create_all was called with our engine
            assert mock_engine == engine
