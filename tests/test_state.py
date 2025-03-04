from pathlib import Path
from unittest.mock import Mock

from velora.state import TrainState, RecordState


class TestTrainState:
    def test_init(self):
        # Default init
        state = TrainState(env="test", total_episodes=100)
        assert state.env == "test"
        assert state.total_episodes == 100
        assert state.status == "start"
        assert state.current_ep == 0
        assert state.avg_reward == 0
        assert state.stop_training is False

        # Custom init
        state = TrainState(
            env="test",
            total_episodes=200,
            status="step",
            current_ep=10,
            avg_reward=15.5,
            stop_training=True,
        )
        assert state.env == "test"
        assert state.total_episodes == 200
        assert state.status == "step"
        assert state.current_ep == 10
        assert state.avg_reward == 15.5
        assert state.stop_training is True

    def test_update(self):
        state = TrainState(env="test", total_episodes=100)

        # Update just status
        state.update(status="step")
        assert state.env == "test"  # Unchanged
        assert state.status == "step"
        assert state.current_ep == 0  # Unchanged
        assert state.avg_reward == 0  # Unchanged

        # Update just episode
        state.update(current_ep=5)
        assert state.env == "test"  # Unchanged
        assert state.status == "step"  # Unchanged
        assert state.current_ep == 5
        assert state.avg_reward == 0  # Unchanged

        # Update just reward
        state.update(avg_reward=10.5)
        assert state.env == "test"  # Unchanged
        assert state.status == "step"  # Unchanged
        assert state.current_ep == 5  # Unchanged
        assert state.avg_reward == 10.5

        # Update all values
        state.update(status="complete", current_ep=10, avg_reward=20.0)
        assert state.env == "test"  # Unchanged
        assert state.status == "complete"
        assert state.current_ep == 10
        assert state.avg_reward == 20.0


class TestRecordState:
    def test_to_wrapper(self):
        # Create mock trigger functions
        episode_trigger = Mock(return_value=True)
        step_trigger = Mock(return_value=False)

        # Create RecordState with both triggers
        state = RecordState(
            dirpath=Path("test/video/path"),
            method="episode",
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
        )

        # Get wrapper parameters
        wrapper_params = state.to_wrapper()

        # Check the wrapper parameters
        assert wrapper_params["video_folder"] == Path("test/video/path")
        assert wrapper_params["episode_trigger"] is episode_trigger
        assert wrapper_params["step_trigger"] is step_trigger

        # Ensure they work as expected
        assert wrapper_params["episode_trigger"]() is True
        assert wrapper_params["step_trigger"]() is False

        # Test with just episode trigger
        state = RecordState(
            dirpath=Path("test/video/path"),
            method="episode",
            episode_trigger=episode_trigger,
        )

        wrapper_params = state.to_wrapper()
        assert wrapper_params["video_folder"] == Path("test/video/path")
        assert wrapper_params["episode_trigger"] is episode_trigger
        assert wrapper_params["step_trigger"] is None

        # Test with just step trigger
        state = RecordState(
            dirpath=Path("test/video/path"),
            method="step",
            step_trigger=step_trigger,
        )

        wrapper_params = state.to_wrapper()
        assert wrapper_params["video_folder"] == Path("test/video/path")
        assert wrapper_params["episode_trigger"] is None
        assert wrapper_params["step_trigger"] is step_trigger
