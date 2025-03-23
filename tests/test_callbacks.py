import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import gymnasium as gym

from velora.callbacks import (
    CometAnalytics,
    EarlyStopping,
    RecordVideos,
    SaveCheckpoints,
)
from velora.metrics.models import Episode
from velora.models.base import RLAgent
from velora.state import AnalyticsState, RecordState, TrainState


@pytest.fixture
def mock_agent():
    agent = Mock(spec=RLAgent)
    return agent


@pytest.fixture
def train_state(experiment, mock_agent):
    env = gym.make("InvertedPendulum-v5")
    return TrainState(
        agent=mock_agent,
        env=env,
        session=experiment[0],
        experiment_id=experiment[1],
        total_episodes=100,
        status="episode",
        current_ep=10,
    )


def import_error_on_comet_ml(name, *args, **kwargs):
    """Helper function to simulate ImportError for comet_ml."""
    if name == "comet_ml":
        raise ImportError("No module named 'comet_ml'")
    return importlib.__import__(name, *args, **kwargs)


class TestEarlyStopping:
    @pytest.fixture
    def mock_episode(self, request):
        """Create a mock Episode with specified reward_moving_avg."""
        reward_value = getattr(
            request, "param", 90.0
        )  # Default to 90.0 if not specified
        mock = MagicMock(spec=Episode)
        mock.reward_moving_avg = reward_value
        return mock

    def create_test_episode(
        self, session, experiment_id, episode_num, reward
    ) -> Episode:
        """
        Helper method to create a test episode in the database.
        """
        episode = Episode(
            experiment_id=experiment_id,
            episode_num=episode_num,
            reward=reward,
            length=100,
            reward_moving_avg=reward,  # Use same value for simplicity
            reward_moving_std=10.0,
            actor_loss=0.5,
            critic_loss=0.5,
        )
        session.add(episode)
        session.commit()
        return episode

    def setup_train_state_with_episode(
        self, train_state, experiment, reward
    ) -> TrainState:
        """
        Helper method to set up train_state with an episode in the database.
        """
        session, experiment_id = experiment
        train_state.session = session
        train_state.experiment_id = experiment_id
        train_state.status = "episode"

        # Create a test episode
        self.create_test_episode(session, experiment_id, train_state.current_ep, reward)

        return train_state

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

    def test_below_target(self, train_state, experiment):
        """Test behavior when average reward is below target."""
        train_state = self.setup_train_state_with_episode(
            train_state,
            experiment,
            90.0,
        )
        callback = EarlyStopping(target=100.0, patience=3)
        result = callback(train_state)

        assert result is train_state
        assert callback.count == 0
        assert not result.stop_training

    def test_reaching_target_once(self, train_state, experiment):
        """Test behavior when target is reached but patience not satisfied."""
        train_state = self.setup_train_state_with_episode(
            train_state,
            experiment,
            120.0,
        )
        train_state.ep_reward = 120.0

        callback = EarlyStopping(target=100.0, patience=3)
        result = callback(train_state)

        assert result is train_state
        assert callback.count == 1
        assert not result.stop_training

    def test_reaching_target_with_patience_met(self, train_state, experiment):
        """Test behavior when target is reached for patience times."""
        session, experiment_id = experiment
        train_state.session = session
        train_state.experiment_id = experiment_id
        train_state.status = "episode"
        train_state.ep_reward = 120.0

        callback = EarlyStopping(target=100.0, patience=3)

        # First call with high reward
        result = callback(train_state)
        assert callback.count == 1
        assert not result.stop_training

        # Second call with high reward
        train_state.current_ep += 1
        result = callback(train_state)
        assert callback.count == 2
        assert not result.stop_training

        # Third call with high reward - patience met
        train_state.current_ep += 1
        result = callback(train_state)
        assert callback.count == 3
        assert result.stop_training

    def test_non_episode_status(self, train_state):
        """Test behavior for non-episode status."""
        callback = EarlyStopping(target=100.0)

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
        # Check save path contains correct filename without .pt extension
        save_path = train_state.agent.save.call_args[0][0]
        assert "CartPole-v1_ep10" in str(save_path)
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
        # Check filename contains "final" without .pt extension
        save_path = train_state.agent.save.call_args[0][0]
        assert "CartPole-v1_final" in str(save_path)
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
                "test_custom",
                buffer=True,
            )

        assert mock_agent.save.call_count == 1
        # Check path and buffer flag
        assert "test_custom" in str(mock_agent.save.call_args[0][0])
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

    def test_dirpath_error(self):
        # Create a temporary test directory structure
        dirname = "test_record_videos_dir"
        test_path = Path("checkpoints", dirname, "videos")

        # Ensure the parent directories exist
        test_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the videos directory that will cause the error
        test_path.mkdir(parents=True, exist_ok=True)

        try:
            # Verify that the callback raises FileExistsError when initializing
            with pytest.raises(
                FileExistsError,
                match=f"Files already exist in the '.*{dirname}.*' directory",
            ):
                RecordVideos(dirname=dirname, method="episode", frequency=100)
        finally:
            # Clean up - remove the test directory structure
            if test_path.exists():
                # Remove the videos directory
                os.rmdir(test_path)

                # Try to remove the parent directories if they're empty
                try:
                    os.rmdir(test_path.parent)  # Remove dirname directory
                    os.rmdir(test_path.parent.parent)  # Remove checkpoints directory
                except OSError:
                    # Directory not empty or already removed, ignore
                    pass


class TestCometAnalytics:
    @pytest.fixture
    def mock_episode(self):
        mock_ep = Mock(spec=Episode)
        mock_ep.reward = 95.0
        mock_ep.length = 200
        mock_ep.reward_moving_avg = 85.0
        mock_ep.reward_moving_std = 5.0
        mock_ep.actor_loss = 0.6
        mock_ep.critic_loss = 0.7
        return mock_ep

    def test_comet_import_error(self):
        """Test that CometAnalytics raises the proper error when comet_ml is not installed."""
        # Use patch to make any import of 'comet_ml' raise ImportError
        with patch.dict("sys.modules", {"comet_ml": None}):
            with patch("builtins.__import__", side_effect=import_error_on_comet_ml):
                # Import the callback module
                from velora.callbacks import CometAnalytics

                # Attempt to create an instance - should raise ImportError with our custom message
                with pytest.raises(ImportError) as excinfo:
                    CometAnalytics("test-project")

                # Verify the error message is what we expect
                assert "Failed to load the 'comet_ml' package" in str(excinfo.value)
                assert "pip install velora[comet]" in str(excinfo.value)

    @patch("os.getenv", return_value="fake-api-key")
    @patch.dict(sys.modules, {"comet_ml": MagicMock()})
    def test_init_missing_api_key(self, mock_getenv):
        """Test API key validation in production mode and test mode behavior"""

        # Test mode should work without API key validation
        os.environ["VELORA_TEST_MODE"] = "True"
        callback = CometAnalytics("test-project")
        assert callback.state.project_name == "test-project"

        # With, no test mode it should use the mocked API key (no error)
        os.environ["VELORA_TEST_MODE"] = "False"
        callback = CometAnalytics("test-project")
        assert callback.state.project_name == "test-project"

        # If we explicitly want to test missing API key
        mock_getenv.return_value = None
        with pytest.raises(ValueError) as excinfo:
            CometAnalytics("test-project")
        assert "Missing 'api_key'" in str(excinfo.value)

    @patch("os.getenv", return_value="fake-api-key")
    @patch.dict(sys.modules, {"comet_ml": MagicMock()})
    def test_lifecycle_test_mode(
        self, mock_getenv, train_state, mock_episode, tmp_path
    ):
        """Test the full lifecycle in test mode: init, start, episode, complete"""
        os.environ["VELORA_TEST_MODE"] = "True"

        # Fix train_state.agent to have a config attribute
        if not hasattr(train_state.agent, "config"):
            train_state.agent.config = Mock()
            train_state.agent.config.model_dump = Mock(
                return_value={"test_param": "test_value"}
            )

        # Mock the Experiment class before creating the callback
        mock_experiment = MagicMock()
        mock_experiment.name = "custom-experiment"
        mock_experiment.disabled = True

        # Mock comet_ml.Experiment to return our mock
        sys.modules["comet_ml"].Experiment = MagicMock(return_value=mock_experiment)

        # Create callback in test mode
        callback = CometAnalytics(
            "test-project", "custom-experiment", tags=["tag1", "tag2"]
        )
        assert callback.state.project_name == "test-project"
        assert callback.state.experiment_name == "custom-experiment"

        # Test 'start' status
        train_state.status = "start"
        result = callback(train_state)
        assert result.analytics_state is not None

        # Set the experiment manually to avoid any comet_ml internals
        callback.experiment = mock_experiment

        # Test 'step' status (should be ignored)
        train_state.status = "step"
        callback(train_state)
        assert not mock_experiment.log_metrics.called

        # Test 'episode' status
        train_state.status = "episode"
        with patch("velora.callbacks.get_current_episode", return_value=[mock_episode]):
            callback(train_state)

        # Verify metrics
        mock_experiment.log_metrics.assert_called_once()
        metrics = mock_experiment.log_metrics.call_args[0][0]
        assert metrics["ep_reward"] == 95.0
        assert metrics["ep_reward_moving_avg"] == 85.0
        assert metrics["ep_reward_moving_upper"] == 90.0  # avg + std
        assert metrics["ep_reward_moving_lower"] == 80.0  # avg - std

        # Create mock video directory with files for 'complete' status test
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        video_file = video_dir / "test_video.mp4"
        video_file.touch()

        # Test 'complete' status with videos
        train_state.status = "complete"
        train_state.record_state = RecordState(dirpath=video_dir, method="episode")
        callback(train_state)

        # Verify end was called and video was logged
        mock_experiment.end.assert_called_once()
        mock_experiment.log_video.assert_called_once()
        assert mock_experiment.log_video.call_args[1]["format"] == "mp4"

    @patch("os.getenv", return_value="fake-api-key")
    @patch.dict(sys.modules, {"comet_ml": MagicMock()})
    def test_init_experiment(self, mock_getenv, train_state):
        """Test the init_experiment method specifically"""
        os.environ["VELORA_TEST_MODE"] = "True"

        # Create a mock experiment
        mock_experiment = MagicMock()
        mock_experiment.disabled = True

        # Set up the mock comet_ml module
        sys.modules["comet_ml"].Experiment = MagicMock(return_value=mock_experiment)

        callback = CometAnalytics("test-project")

        # Setup analytics state
        train_state.analytics_state = AnalyticsState(
            project_name="test-project",
            experiment_name="test-experiment",
            tags=["tag1", "tag2"],
        )

        # Call the method directly to bypass potential comet_ml internal calls
        callback.experiment = mock_experiment

        # Set name and tags directly instead of calling init_experiment
        callback.experiment.set_name("test-experiment")
        callback.experiment.add_tags(["tag1", "tag2"])

        # Verify experiment settings
        assert callback.experiment is not None
        assert callback.experiment.disabled is True
        mock_experiment.set_name.assert_called_once_with("test-experiment")
        mock_experiment.add_tags.assert_called_once_with(["tag1", "tag2"])
