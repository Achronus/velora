import pytest
from unittest.mock import patch

from velora.metrics.tracker import TrainMetrics, MetricsTracker


class TestTrainMetrics:
    def test_default_init(self):
        metrics = TrainMetrics()

        assert hasattr(metrics, "ep_rewards")
        assert hasattr(metrics, "critic_losses")
        assert hasattr(metrics, "actor_losses")

        assert isinstance(metrics.ep_rewards, list)
        assert isinstance(metrics.critic_losses, list)
        assert isinstance(metrics.actor_losses, list)

        assert len(metrics.ep_rewards) == 0
        assert len(metrics.critic_losses) == 0
        assert len(metrics.actor_losses) == 0

    def test_init_custom(self):
        ep_rewards = [10.0, 20.0, 30.0]
        critic_losses = [1.5, 1.2, 0.9]
        actor_losses = [0.8, 0.7, 0.6]

        metrics = TrainMetrics(
            ep_rewards=ep_rewards,
            critic_losses=critic_losses,
            actor_losses=actor_losses,
        )

        assert len(metrics.ep_rewards) == len(ep_rewards)
        assert all(a == b for a, b in zip(metrics.ep_rewards, ep_rewards))

        assert len(metrics.critic_losses) == len(critic_losses)
        assert all(a == b for a, b in zip(metrics.critic_losses, critic_losses))

        assert len(metrics.actor_losses) == len(actor_losses)
        assert all(a == b for a, b in zip(metrics.actor_losses, actor_losses))

    def test_mutability(self):
        metrics = TrainMetrics()

        metrics.ep_rewards.append(10.0)
        metrics.critic_losses.append(1.5)
        metrics.actor_losses.append(0.8)

        assert metrics.ep_rewards == [10.0]
        assert metrics.critic_losses == [1.5]
        assert metrics.actor_losses == [0.8]


class TestMetricsTracker:
    @pytest.fixture
    def tracker(self) -> MetricsTracker:
        return MetricsTracker(total_episodes=100, window_size=10)

    def test_init(self, tracker: MetricsTracker):
        assert tracker.total_episodes == 100
        assert tracker.window_size == 10
        assert isinstance(tracker.storage, TrainMetrics)
        assert len(tracker.storage.ep_rewards) == 0
        assert len(tracker.storage.critic_losses) == 0
        assert len(tracker.storage.actor_losses) == 0

    def test_log_single_metric(self, tracker: MetricsTracker):
        tracker.log({"ep_rewards": 15.0})

        assert len(tracker.storage.ep_rewards) == 1
        assert tracker.storage.ep_rewards[0] == 15.0
        assert len(tracker.storage.critic_losses) == 0
        assert len(tracker.storage.actor_losses) == 0

    def test_log_multiple_metrics(self, tracker: MetricsTracker):
        tracker.log({"ep_rewards": 15.0, "critic_losses": 1.2, "actor_losses": 0.7})

        assert len(tracker.storage.ep_rewards) == 1
        assert tracker.storage.ep_rewards[0] == 15.0
        assert len(tracker.storage.critic_losses) == 1
        assert tracker.storage.critic_losses[0] == 1.2
        assert len(tracker.storage.actor_losses) == 1
        assert tracker.storage.actor_losses[0] == 0.7

    def test_log_sequential_calls(self, tracker: MetricsTracker):
        tracker.log({"ep_rewards": 15.0})
        tracker.log({"critic_losses": 1.2})
        tracker.log({"actor_losses": 0.7})
        tracker.log({"ep_rewards": 20.0})

        assert tracker.storage.ep_rewards == [15.0, 20.0]
        assert tracker.storage.critic_losses == [1.2]
        assert tracker.storage.actor_losses == [0.7]

    def test_avg_reward_empty(self, tracker: MetricsTracker):
        assert tracker.avg_reward() == 0

    def test_avg_reward_less_than_window(self, tracker: MetricsTracker):
        # Add 5 rewards (less than window_size=10)
        for i in range(5):
            tracker.log({"ep_rewards": float(i * 10)})

        # Should return 0 since less than window_size
        assert tracker.avg_reward() == 0

    def test_avg_reward_with_full_window(self, tracker: MetricsTracker):
        # Add 10 rewards (equal to window_size=10)
        for i in range(10):
            tracker.log({"ep_rewards": float(i * 10)})

        # Expected average: (0 + 10 + 20 + ... + 90) / 10 = 45.0
        assert tracker.avg_reward() == 45.0

    def test_avg_reward_with_more_than_window(self, tracker: MetricsTracker):
        # Add 15 rewards (more than window_size=10)
        for i in range(15):
            tracker.log({"ep_rewards": float(i * 10)})

        # Should only consider last 10 values: (50 + 60 + ... + 140) / 10 = 95.0
        assert tracker.avg_reward() == 95.0

    @patch("numpy.mean")
    def test_avg_reward_uses_numpy_mean(self, mock_mean, tracker: MetricsTracker):
        mock_mean.return_value = 42.0

        # Add enough rewards to trigger mean calculation
        for i in range(10):
            tracker.log({"ep_rewards": float(i)})

        result = tracker.avg_reward()

        # Verify numpy.mean was called with the right data
        mock_mean.assert_called_once()
        assert result == 42.0

    @patch("builtins.print")
    def test_print_method(self, mock_print, tracker: MetricsTracker):
        # Add some data
        for i in range(10):
            tracker.log(
                {
                    "ep_rewards": float(i * 10),
                    "critic_losses": float(i),
                    "actor_losses": float(i) / 10,
                }
            )

        # Call print with current episode
        current_ep = 50
        tracker.print(current_ep)

        # Verify print was called with expected format
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]

        # Check that the string contains expected values
        assert f"Episode: {current_ep}/{tracker.total_episodes}" in call_args
        assert "Avg Reward: 45.00" in call_args
        assert "Critic Loss: 4.50" in call_args
        assert "Actor Loss: 0.45" in call_args

    def test_print_method_calculations(self, tracker: MetricsTracker):
        # Add enough rewards to trigger avg_reward calculation
        for i in range(15):
            tracker.log({"ep_rewards": 42.0})
            tracker.log({"critic_losses": 2.5})
            tracker.log({"actor_losses": 1.5})

        # Patch print to capture output
        with patch("builtins.print") as mock_print:
            tracker.print(50)

            call_args = mock_print.call_args[0][0]
            # Check values are included in output
            assert "Avg Reward: 42.00" in call_args
            assert "Critic Loss: 2.50" in call_args
            assert "Actor Loss: 1.50" in call_args

    def test_log_with_invalid_key(self, tracker: MetricsTracker):
        with pytest.raises(AttributeError):
            tracker.log({"invalid_key": 10.0})

    def test_storage_property(self, tracker: MetricsTracker):
        assert isinstance(tracker.storage, TrainMetrics)

        # Add some data
        tracker.log({"ep_rewards": 10.0})
        assert tracker.storage.ep_rewards == [10.0]

        # Verify it's the same instance
        storage = tracker.storage
        tracker.log({"ep_rewards": 20.0})
        assert storage.ep_rewards == [10.0, 20.0]
