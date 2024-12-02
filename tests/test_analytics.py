from unittest.mock import MagicMock, patch
import pytest

from velora.analytics.base import NullAnalytics
from velora.analytics.wandb import WeightsAndBiases
from velora.exc import RunNotFoundError


class TestNullAnalytics:
    @pytest.fixture
    def analytics(self) -> NullAnalytics:
        return NullAnalytics()

    @staticmethod
    def test_init(analytics: NullAnalytics):
        run = analytics.init("test1")
        assert isinstance(run, NullAnalytics)


class TestWeightsAndBiases:
    @pytest.fixture
    def analytics(self) -> WeightsAndBiases:
        """Fixture to create a WeightsAndBiases instance with wandb.login mocked."""
        with patch("wandb.login"):
            return WeightsAndBiases()

    @pytest.fixture
    def mock_run(self, analytics: WeightsAndBiases) -> MagicMock:
        """Fixture to create a mock run and attach it to the analytics instance."""
        mock_run = MagicMock()
        analytics._run = mock_run
        return mock_run

    @staticmethod
    def test_model_post_init():
        with patch("wandb.login") as mock_login:
            _ = WeightsAndBiases()
            mock_login.assert_called_once()

    @staticmethod
    def test_init_method(analytics: WeightsAndBiases):
        with patch("wandb.init") as mock_init:
            mock_run = MagicMock()
            mock_init.return_value = mock_run

            project_name = "test_project"
            run_name = "test_run"
            config = {"learning_rate": 0.01, "batch_size": 32}

            analytics.init(project_name, run_name, config)

            mock_init.assert_called_once_with(
                project=project_name, name=run_name, config=config
            )
            assert analytics.run == mock_run

    @staticmethod
    def test_log_method_no_run(analytics: WeightsAndBiases):
        metrics = {"accuracy": 0.95}

        with pytest.raises(RunNotFoundError, match="No run instance found"):
            analytics.log(metrics)

    @staticmethod
    def test_finish_method_no_run(analytics: WeightsAndBiases):
        with pytest.raises(RunNotFoundError, match="No run instance found"):
            analytics.finish()
