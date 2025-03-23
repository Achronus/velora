import pytest
from unittest.mock import Mock, patch

from velora.buffer.replay import ReplayBuffer
from velora.training.display import training_info
from velora.models.base import RLAgent
from velora.callbacks import TrainCallback


class TestTrainingInfo:
    """Test suite for the training_info function."""

    @pytest.fixture
    def mock_agent(self):
        """Fixture for creating a mock agent."""
        buffer = Mock(spec=ReplayBuffer)
        agent = Mock(spec=RLAgent)
        agent.__class__.__name__ = "MockAgent"
        buffer.__class__.__name__ = "MockBuffer"

        agent.active_params = 5000
        agent.total_params = 10000
        agent.buffer = buffer
        agent.buffer.capacity = 100000
        return agent

    @pytest.fixture
    def mock_callbacks(self):
        """Fixture for creating mock callbacks."""
        cb1 = Mock(spec=TrainCallback)
        cb1.info.return_value = "Callback 1 info"
        cb2 = Mock(spec=TrainCallback)
        cb2.info.return_value = "Callback 2 info"
        return [cb1, cb2]

    @patch("builtins.print")
    def test_basic(self, mock_print, mock_agent):
        """Test the basic functionality with no callbacks."""
        training_info(
            agent=mock_agent,
            env_id="TestEnv-v0",
            n_episodes=1000,
            batch_size=64,
            window_size=100,
            callbacks=[],
            device="cpu",
        )

        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]

        # Verify basic information is in the output
        assert "Training 'MockAgent' agent on 'TestEnv-v0'" in printed_output
        assert "Using 'MockBuffer' with 'capacity=100K'" in printed_output
        assert "batch_size=64" in printed_output
        assert "device 'cpu'" in printed_output
        assert "window_size=100" in printed_output
        assert "5,000/10,000" in printed_output
        assert "Active Callbacks:" not in printed_output  # No callbacks

    @patch("builtins.print")
    def test_with_callbacks(self, mock_print, mock_agent, mock_callbacks):
        """Test with callbacks."""
        training_info(
            agent=mock_agent,
            env_id="TestEnv-v0",
            n_episodes=1000,
            batch_size=64,
            window_size=100,
            callbacks=mock_callbacks,
            device="cpu",
        )

        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]

        # Verify callback information is included
        assert "Active Callbacks:" in printed_output
        assert "Callback 1 info" in printed_output
        assert "Callback 2 info" in printed_output

    @patch("builtins.print")
    def test_large_parameter_count(self, mock_print, mock_agent):
        """Test with large parameter counts."""
        mock_agent.active_params = 12_345_678
        mock_agent.total_params = 20_000_000

        training_info(
            agent=mock_agent,
            env_id="TestEnv-v0",
            n_episodes=1000,
            batch_size=64,
            window_size=100,
            callbacks=[],
            device="cuda",
        )

        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]

        # Verify parameter formatting for large numbers
        assert "12.35M/20M" in printed_output or "12.35M/20.0M" in printed_output

    @patch("builtins.print")
    def test_large_episode_count(self, mock_print, mock_agent):
        """Test with large episode count."""
        training_info(
            agent=mock_agent,
            env_id="TestEnv-v0",
            n_episodes=1_000_000,
            batch_size=64,
            window_size=100,
            callbacks=[],
            device="cpu",
        )

        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]

        # Verify episode count formatting
        assert "1M" in printed_output

    @patch("builtins.print")
    def test_gpu_device(self, mock_print, mock_agent):
        """Test with GPU device."""
        training_info(
            agent=mock_agent,
            env_id="TestEnv-v0",
            n_episodes=1000,
            batch_size=64,
            window_size=100,
            callbacks=[],
            device="cuda:0",
        )

        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]

        # Verify device information
        assert "device 'cuda:0'" in printed_output
