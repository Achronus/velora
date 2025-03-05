import pytest
from unittest.mock import patch

from velora.training.metrics import MetricStorage, TrainMetrics


class TestTrainMetrics:
    @pytest.fixture
    def tracker(self) -> TrainMetrics:
        return TrainMetrics(n_episodes=100, window_size=10)

    def test_init(self, tracker: TrainMetrics):
        assert tracker.n_episodes == 100
        assert tracker.window_size == 10
        assert isinstance(tracker.storage, MetricStorage)
        assert len(tracker.storage.ep_rewards) == 0
        assert len(tracker.storage.critic_losses) == 0
        assert len(tracker.storage.actor_losses) == 0

    def test_add_single_metric(self, tracker: TrainMetrics):
        tracker.add({"ep_rewards": 15.0})

        assert len(tracker.storage.ep_rewards) == 1
        assert tracker.storage.ep_rewards[0] == 15.0
        assert len(tracker.storage.critic_losses) == 0
        assert len(tracker.storage.actor_losses) == 0

    def test_add_multiple_metrics(self, tracker: TrainMetrics):
        tracker.add({"ep_rewards": 15.0, "critic_losses": 1.2, "actor_losses": 0.7})

        assert len(tracker.storage.ep_rewards) == 1
        assert tracker.storage.ep_rewards[0] == 15.0
        assert len(tracker.storage.critic_losses) == 1
        assert tracker.storage.critic_losses[0] == 1.2
        assert len(tracker.storage.actor_losses) == 1
        assert tracker.storage.actor_losses[0] == 0.7

    def test_add_sequential_calls(self, tracker: TrainMetrics):
        tracker.add({"ep_rewards": 15.0})
        tracker.add({"critic_losses": 1.2})
        tracker.add({"actor_losses": 0.7})
        tracker.add({"ep_rewards": 20.0})

        assert tracker.storage.ep_rewards == [15.0, 20.0]
        assert tracker.storage.critic_losses == [1.2]
        assert tracker.storage.actor_losses == [0.7]

    def test_avg_reward_empty(self, tracker: TrainMetrics):
        assert tracker.avg_reward() == 0

    def test_avg_reward_less_than_window(self, tracker: TrainMetrics):
        # Add 5 rewards (less than window_size=10)
        for i in range(5):
            tracker.add({"ep_rewards": float(i * 10)})

        # Should return 0 since less than window_size
        assert tracker.avg_reward() == 0

    def test_avg_reward_with_full_window(self, tracker: TrainMetrics):
        # Add 10 rewards (equal to window_size=10)
        for i in range(10):
            tracker.add({"ep_rewards": float(i * 10)})

        # Expected average: (0 + 10 + 20 + ... + 90) / 10 = 45.0
        assert tracker.avg_reward() == 45.0

    def test_avg_reward_with_more_than_window(self, tracker: TrainMetrics):
        # Add 15 rewards (more than window_size=10)
        for i in range(15):
            tracker.add({"ep_rewards": float(i * 10)})

        # Should only consider last 10 values: (50 + 60 + ... + 140) / 10 = 95.0
        assert tracker.avg_reward() == 95.0

    @patch("numpy.mean")
    def test_avg_reward_uses_numpy_mean(self, mock_mean, tracker: TrainMetrics):
        mock_mean.return_value = 42.0

        # Add enough rewards to trigger mean calculation
        for i in range(10):
            tracker.add({"ep_rewards": float(i)})

        result = tracker.avg_reward()

        # Verify numpy.mean was called with the right data
        mock_mean.assert_called_once()
        assert result == 42.0

    @patch("builtins.print")
    def test_print_method(self, mock_print, tracker: TrainMetrics):
        # Add some data
        for i in range(10):
            tracker.add(
                {
                    "ep_rewards": float(i * 10),
                    "critic_losses": float(i),
                    "actor_losses": float(i) / 10,
                }
            )

        # Call print with current episode
        current_ep = 50
        tracker.info(current_ep)

        # Verify print was called with expected format
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]

        # Check that the string contains expected values
        assert f"Episode: {current_ep}/{tracker.n_episodes}" in call_args
        assert "Avg Reward: 45.00" in call_args
        assert "Critic Loss: 4.50" in call_args
        assert "Actor Loss: 0.45" in call_args

    def test_print_method_calculations(self, tracker: TrainMetrics):
        # Add enough rewards to trigger avg_reward calculation
        for i in range(15):
            tracker.add({"ep_rewards": 42.0})
            tracker.add({"critic_losses": 2.5})
            tracker.add({"actor_losses": 1.5})

        # Patch print to capture output
        with patch("builtins.print") as mock_print:
            tracker.info(50)

            call_args = mock_print.call_args[0][0]
            # Check values are included in output
            assert "Avg Reward: 42.00" in call_args
            assert "Critic Loss: 2.50" in call_args
            assert "Actor Loss: 1.50" in call_args

    def test_log_with_invalid_key(self, tracker: TrainMetrics):
        with pytest.raises(ValueError):
            tracker.add({"invalid_key": 10.0})

    def test_storage_property(self, tracker: TrainMetrics):
        assert isinstance(tracker.storage, MetricStorage)

        # Add some data
        tracker.add({"ep_rewards": 10.0})
        assert tracker.storage.ep_rewards == [10.0]

        # Verify it's the same instance
        storage = tracker.storage
        tracker.add({"ep_rewards": 20.0})
        assert storage.ep_rewards == [10.0, 20.0]
