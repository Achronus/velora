import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import gymnasium as gym

from velora.callbacks import EarlyStopping, RecordVideos, SaveCheckpoints
from velora.models.base import RLAgent
from velora.state import TrainState


@pytest.fixture
def mock_agent():
    agent = Mock(spec=RLAgent)
    return agent


@pytest.fixture
def mock_metrics():
    metrics = Mock()
    return metrics


@pytest.fixture
def train_state(mock_agent, mock_metrics):
    env = gym.make("CartPole-v1")
    return TrainState(
        agent=mock_agent,
        env=env,
        total_episodes=100,
        metrics=mock_metrics,
        status="episode",
        current_ep=10,
    )


class TestEarlyStopping:
    def test_init(self):
        # Default initialization
        callback = EarlyStopping(target=100.0)
        assert callback.target == 100.0
        assert callback.patience == 3
        assert callback.count == 0

        # Custom initialization
        callback = EarlyStopping(target=50.0, patience=5)
        assert callback.target == 50.0
        assert callback.patience == 5
        assert callback.count == 0

    def test_below_target(self, train_state):
        """Test behavior when average reward is below target."""
        callback = EarlyStopping(target=100.0, patience=3)

        # Mock the metrics.avg_reward() method to return a value below target
        train_state.metrics.avg_reward.return_value = 90.0

        result = callback(train_state)

        assert result is train_state
        assert callback.count == 0
        assert not result.stop_training

    def test_reaching_target_once(self, train_state):
        """Test behavior when target is reached but patience not satisfied."""
        callback = EarlyStopping(target=100.0, patience=3)

        # Mock the metrics.avg_reward() method to return a value above target
        train_state.metrics.avg_reward.return_value = 105.0

        result = callback(train_state)

        assert result is train_state
        assert callback.count == 1
        assert not result.stop_training

    def test_reaching_target_with_patience_met(self, train_state):
        """Test behavior when target is reached for patience times."""
        callback = EarlyStopping(target=100.0, patience=3)

        # Mock the metrics.avg_reward() method to return a value above target
        train_state.metrics.avg_reward.return_value = 105.0

        # First call
        callback(train_state)
        assert callback.count == 1

        # Second call
        callback(train_state)
        assert callback.count == 2

        # Third call - patience met
        result = callback(train_state)
        assert callback.count == 3
        assert result.stop_training

    def test_inconsistent_rewards(self, train_state):
        """Test behavior when rewards fluctuate around target."""
        callback = EarlyStopping(target=100.0, patience=2)

        # First call - above target
        train_state.metrics.avg_reward.return_value = 110.0
        callback(train_state)
        assert callback.count == 1

        # Second call - below target, resets counter
        train_state.metrics.avg_reward.return_value = 90.0
        callback(train_state)
        assert callback.count == 0

        # Third call - above target again
        train_state.metrics.avg_reward.return_value = 105.0
        callback(train_state)
        assert callback.count == 1

        # Fourth call - above target, patience met
        train_state.metrics.avg_reward.return_value = 105.0
        result = callback(train_state)
        assert callback.count == 2
        assert result.stop_training

    def test_non_episode_status(self, train_state):
        """Test behavior for non-episode status."""
        callback = EarlyStopping(target=100.0)
        train_state.metrics.avg_reward.return_value = 110.0

        # Step status should be ignored
        train_state.status = "step"
        result = callback(train_state)
        assert callback.count == 0
        assert not result.stop_training

        # Complete status should be ignored
        train_state.status = "complete"
        result = callback(train_state)
        assert callback.count == 0
        assert not result.stop_training

    def test_already_stopped(self, train_state):
        """Test behavior when training is already stopped."""
        callback = EarlyStopping(target=100.0)
        train_state.metrics.avg_reward.return_value = 110.0
        train_state.stop_training = True

        result = callback(train_state)

        assert result is train_state
        assert callback.count == 0  # No increment
        assert result.stop_training  # Still stopped


class TestSaveCheckpoints:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_init(self):
        # Default initialization
        callback = SaveCheckpoints("model_dir")
        assert callback.filepath == Path("checkpoints", "model_dir", "saves")
        assert callback.frequency == 100
        assert callback.buffer is False

        # Custom initialization
        callback = SaveCheckpoints("custom", frequency=50, buffer=True)
        assert callback.filepath == Path("checkpoints", "custom", "saves")
        assert callback.frequency == 50
        assert callback.buffer is True

    def test_non_episode_status(self, train_state):
        """Test behavior for non-episode status that should be ignored."""
        callback = SaveCheckpoints("model_dir")
        train_state.status = "step"

        # Create directory patch to verify no directory is created
        with patch.object(Path, "mkdir") as mock_mkdir:
            result = callback(train_state)
            assert mock_mkdir.call_count == 0
            assert train_state.agent.save.call_count == 0

        assert result is train_state

    @patch("pathlib.Path.mkdir")
    def test_save_on_frequency(self, mock_mkdir, train_state):
        """Test saving checkpoint at specified frequency."""
        callback = SaveCheckpoints("model_dir", frequency=10)
        train_state.env.spec.name = "CartPole-v1"  # Set environment name
        train_state.current_ep = 10  # Episode matches frequency exactly

        result = callback(train_state)

        assert mock_mkdir.call_count == 1
        assert train_state.agent.save.call_count == 1
        # Check filename contains episode number
        save_path = train_state.agent.save.call_args[0][0]
        assert "CartPole-v1_ep10.pt" in str(save_path)
        assert train_state.agent.save.call_args[1]["buffer"] is False

        assert result is train_state

    @patch("pathlib.Path.mkdir")
    def test_no_save_between_frequency(self, mock_mkdir, train_state):
        """Test no save between frequency intervals."""
        callback = SaveCheckpoints("model_dir", frequency=10)
        train_state.current_ep = 11  # Episode doesn't match frequency

        result = callback(train_state)

        assert mock_mkdir.call_count == 1  # Still creates directory
        assert train_state.agent.save.call_count == 0
        assert result is train_state

    @patch("pathlib.Path.mkdir")
    def test_save_on_complete(self, mock_mkdir, train_state):
        """Test save on completion."""
        callback = SaveCheckpoints("model_dir", buffer=True)
        train_state.status = "complete"
        train_state.current_ep = 100
        train_state.env.spec.name = "CartPole-v1"  # Set environment name

        result = callback(train_state)

        assert mock_mkdir.call_count == 1
        assert train_state.agent.save.call_count == 1
        # Check filename contains "final"
        save_path = train_state.agent.save.call_args[0][0]
        assert "CartPole-v1_final.pt" in str(save_path)
        # Buffer should be saved
        assert train_state.agent.save.call_args[1]["buffer"] is True

        assert result is train_state

    def test_directory_creation(self, train_state, temp_dir):
        """Test directory creation."""
        model_dir = os.path.join(temp_dir, "test_model")
        save_path = Path(temp_dir, "test_model", "saves")

        # Directory shouldn't exist initially
        assert not os.path.exists(save_path)

        # Create callback with modified filepath for testing
        callback = SaveCheckpoints("model_dir")
        callback.filepath = save_path

        # Call with episode status to trigger directory creation
        train_state.status = "episode"
        train_state.current_ep = 10
        callback(train_state)

        # Directory should be created
        assert os.path.exists(save_path)

    def test_save_checkpoint_method(self, mock_agent):
        """Test the save_checkpoint method directly."""
        callback = SaveCheckpoints("model_dir")

        # Mock the print function to avoid output
        with patch("builtins.print"):
            callback.save_checkpoint(
                mock_agent,
                ep=10,
                filename="test_custom",
                buffer=True,
            )

        assert mock_agent.save.call_count == 1
        # Check path and buffer flag
        assert "test_custom.pt" in str(mock_agent.save.call_args[0][0])
        assert mock_agent.save.call_args[1]["buffer"] is True

    def test_dirname_exists_error(self):
        dirname = "test_model_dir"
        save_path = Path("checkpoints", dirname, "saves")

        try:
            save_path.mkdir(parents=True, exist_ok=True)

            with pytest.raises(FileExistsError):
                SaveCheckpoints(dirname)

        finally:
            # Clean up
            if os.path.exists(save_path.parent):
                shutil.rmtree(save_path.parent)


class TestRecordVideos:
    def test_init(self):
        # Test episode method
        callback = RecordVideos(method="episode", dirname="model_dir")
        assert callback.method == "episode"
        assert callback.dirpath == Path("checkpoints", "model_dir", "videos")
        assert callback.details.method == "episode"
        assert callback.details.episode_trigger is not None
        assert callback.details.step_trigger is None

        # Test step method
        callback = RecordVideos(method="step", dirname="model_dir")
        assert callback.method == "step"
        assert callback.dirpath == Path("checkpoints", "model_dir", "videos")
        assert callback.details.method == "step"
        assert callback.details.episode_trigger is None
        assert callback.details.step_trigger is not None

        # Test custom frequency
        callback = RecordVideos(method="episode", dirname="model_dir", frequency=50)
        assert callback.details.episode_trigger(0) is False  # Skip first item
        assert callback.details.episode_trigger(50) is True
        assert callback.details.episode_trigger(51) is False
        assert callback.details.episode_trigger(100) is True

    def test_invalid_method(self):
        with pytest.raises(ValueError) as excinfo:
            RecordVideos(method="invalid", dirname="model_dir")
        assert "'method='invalid'" in str(excinfo.value)
        assert "Choices: '('episode', 'step')'" in str(excinfo.value)

    def test_call_start_status(self, train_state):
        callback = RecordVideos(method="episode", dirname="model_dir")
        train_state.status = "start"

        with patch("gymnasium.wrappers.RecordVideo") as mock_record_video:
            result = callback(train_state)

            assert result is train_state
            assert result.record_state is not None
            assert result.record_state.method == "episode"
            assert result.record_state.dirpath == callback.dirpath
            mock_record_video.assert_called_once()

    def test_call_other_status(self, train_state):
        """Test callback behavior for non-start statuses."""
        callback = RecordVideos(method="episode", dirname="model_dir")

        # Test with episode status
        train_state.status = "episode"
        train_state.record_state = None
        result = callback(train_state)
        assert result is train_state
        assert result.record_state is None  # Should not be modified

        # Test with step status
        train_state.status = "step"
        result = callback(train_state)
        assert result is train_state
        assert result.record_state is None  # Should not be modified

        # Test with complete status
        train_state.status = "complete"
        result = callback(train_state)
        assert result is train_state
        assert result.record_state is None  # Should not be modified

    def test_trigger_functions(self):
        """Test the trigger functions created for recording."""
        # Episode method with frequency 10
        callback = RecordVideos(method="episode", dirname="model_dir", frequency=10)
        trigger = callback.details.episode_trigger

        assert trigger(0) is False  # First episode not recorded
        assert trigger(5) is False  # Not divisible by frequency
        assert trigger(10) is True  # Divisible by frequency
        assert trigger(20) is True  # Divisible by frequency

        # Step method with frequency 5
        callback = RecordVideos(method="step", dirname="model_dir", frequency=5)
        trigger = callback.details.step_trigger

        assert trigger(0) is False  # First step not recorded
        assert trigger(2) is False  # Not divisible by frequency
        assert trigger(5) is True  # Divisible by frequency
        assert trigger(10) is True  # Divisible by frequency
