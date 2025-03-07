import pytest
from collections import deque

import numpy as np
import torch

from velora.training.metrics import (
    StepStorage,
    MovingMetric,
    SimpleMetricStorage,
    MetricStorage,
    TrainMetrics,
)


class TestStepStorage:
    @pytest.fixture
    def storage(self) -> StepStorage:
        return StepStorage()

    def test_init(self, storage: StepStorage):
        assert storage.critic_losses == []
        assert storage.actor_losses == []

    def test_critic_avg_empty(self, storage: StepStorage):
        assert storage.critic_avg() == 0

    def test_critic_avg_with_values(self, storage: StepStorage):
        storage.critic_losses = [1.0, 2.0, 3.0]
        assert storage.critic_avg() == 2.0

    def test_actor_avg_empty(self, storage: StepStorage):
        assert storage.actor_avg() == 0

    def test_actor_avg_with_values(self, storage: StepStorage):
        storage.actor_losses = [1.0, 2.0, 3.0]
        assert storage.actor_avg() == 2.0

    def test_add_critic_losses(self, storage: StepStorage):
        storage.add({"critic_losses": 2.5})
        assert storage.critic_losses == [2.5]

    def test_add_actor_losses(self, storage: StepStorage):
        storage.add({"actor_losses": 3.5})
        assert storage.actor_losses == [3.5]

    def test_add_both_losses(self, storage: StepStorage):
        storage.add({"critic_losses": 2.5, "actor_losses": 3.5})
        assert storage.critic_losses == [2.5]
        assert storage.actor_losses == [3.5]

    def test_empty(self, storage: StepStorage):
        storage.critic_losses = [1.0, 2.0]
        storage.actor_losses = [3.0, 4.0]
        storage.empty()

        assert storage.critic_losses == []
        assert storage.actor_losses == []


class TestMovingMetric:
    @pytest.fixture
    def metric(self) -> MovingMetric:
        return MovingMetric()

    def test_init(self, metric: MovingMetric):
        assert metric.values == []
        assert isinstance(metric.window, deque)
        assert metric.window_size == 100

    def test_init_with_custom_window_size(self):
        metric = MovingMetric(window_size=50)
        assert metric.window_size == 50

    def test_add(self, metric: MovingMetric):
        metric.add(1.0)

        assert metric.values == [1.0]
        assert list(metric.window) == [1.0]

    def test_add_exceeds_window_size(self):
        metric = MovingMetric(window_size=2)
        metric.add(1.0)
        metric.add(2.0)
        metric.add(3.0)

        assert metric.values == [1.0, 2.0, 3.0]
        assert list(metric.window) == [2.0, 3.0]

    def test_mean_empty(self, metric: MovingMetric):
        assert metric.mean() == 0.0

    def test_mean_with_values(self, metric: MovingMetric):
        metric.window = deque([1.0, 2.0, 3.0])
        assert metric.mean() == 2.0

    def test_mean_with_custom_values(self, metric: MovingMetric):
        assert metric.mean([4.0, 5.0, 6.0]) == 5.0

    def test_std_empty(self, metric: MovingMetric):
        assert metric.std() == 0.0

    def test_std_window_single_value(self, metric: MovingMetric):
        metric.window = deque([1.0])
        assert metric.std() == 0.0

    def test_std_window(self, metric: MovingMetric):
        metric.window = deque([1.0, 2.0, 3.0])
        expected_std = np.std([1.0, 2.0, 3.0]).item()

        assert abs(metric.std() - expected_std) < 1e-6

    def test_std_with_custom_values(self, metric: MovingMetric):
        values = [4.0, 5.0, 6.0]
        expected_std = np.std(values).item()
        assert abs(metric.std(values) - expected_std) < 1e-6

    def test_len(self, metric: MovingMetric):
        assert len(metric) == 0

        metric.values = [1.0, 2.0, 3.0]
        assert len(metric) == 3


class TestSimpleMetricStorage:
    @pytest.fixture
    def storage(self) -> SimpleMetricStorage:
        return SimpleMetricStorage()

    def test_init(self, storage: SimpleMetricStorage):
        assert storage.ep_rewards == []
        assert storage.ep_lengths == []
        assert storage.critic_losses == []
        assert storage.actor_losses == []

    def test_load(self, tmp_path):
        # Create a temporary file with test data
        filepath = tmp_path / "test_metrics.pt"
        test_data = {
            "ep_rewards": [1.0, 2.0],
            "ep_lengths": [10, 20],
            "critic_losses": [0.1, 0.2],
            "actor_losses": [0.3, 0.4],
        }
        torch.save(test_data, filepath)

        # Load the data
        storage = SimpleMetricStorage.load(filepath)

        # Check if the data was loaded correctly
        assert storage.ep_rewards == [1.0, 2.0]
        assert storage.ep_lengths == [10, 20]
        assert storage.critic_losses == [0.1, 0.2]
        assert storage.actor_losses == [0.3, 0.4]


class TestMetricStorage:
    @pytest.fixture
    def storage(self) -> MetricStorage:
        return MetricStorage(50)

    def test_init(self, storage: MetricStorage):
        assert storage._ep_rewards.window_size == 50
        assert storage._ep_lengths.window_size == 50
        assert storage._critic_losses.window_size == 50
        assert storage._actor_losses.window_size == 50
        assert isinstance(storage._current_losses, StepStorage)

    def test_properties(self, storage: MetricStorage):
        assert storage.ep_rewards is storage._ep_rewards
        assert storage.ep_lengths is storage._ep_lengths
        assert storage.critic_losses is storage._critic_losses
        assert storage.actor_losses is storage._actor_losses

    def test_save_state(self, storage: MetricStorage):
        # Add some test data
        storage._ep_rewards.values = [1.0, 2.0]
        storage._ep_lengths.values = [10, 20]
        storage._critic_losses.values = [0.1, 0.2]
        storage._actor_losses.values = [0.3, 0.4]

        state_dict = storage.save_state()

        assert state_dict["ep_rewards"] == [1.0, 2.0]
        assert state_dict["ep_lengths"] == [10, 20]
        assert state_dict["critic_losses"] == [0.1, 0.2]
        assert state_dict["actor_losses"] == [0.3, 0.4]


class TestTrainMetrics:
    @pytest.fixture
    def metrics(self) -> TrainMetrics:
        return TrainMetrics(50, 100)

    def test_init(self, metrics: TrainMetrics):
        assert metrics.window_size == 50
        assert metrics.n_episodes == 100
        assert isinstance(metrics._storage, MetricStorage)
        assert metrics._storage._ep_rewards.window_size == 50

    def test_n_stored(self, metrics: TrainMetrics):
        assert metrics.n_stored == 0

        # Add some episodes
        metrics._storage._ep_rewards.values = [1.0, 2.0, 3.0]
        assert metrics.n_stored == 3

    def test_storage(self, metrics: TrainMetrics):
        assert metrics.storage is metrics._storage

    def test_add_step(self, metrics: TrainMetrics):
        metrics.add_step(0.1, 0.2)
        assert metrics._storage._current_losses.critic_losses == [0.1]
        assert metrics._storage._current_losses.actor_losses == [0.2]

    def test_add_episode(self, metrics: TrainMetrics):
        # Add some step data first
        metrics._storage._current_losses.critic_losses = [0.1, 0.2]
        metrics._storage._current_losses.actor_losses = [0.3, 0.4]

        # Add an episode
        metrics.add_episode(reward=1.5, n_steps=30)

        # Check that episode metrics were added
        assert metrics._storage._ep_rewards.values == [1.5]
        assert metrics._storage._ep_lengths.values == [30]
        assert round(metrics._storage._critic_losses.values[0], 2) == 0.15
        assert round(metrics._storage._actor_losses.values[0], 2) == 0.35

        # Check that step storage was emptied
        assert metrics._storage._current_losses.critic_losses == []
        assert metrics._storage._current_losses.actor_losses == []

    def test_avg_reward(self, metrics: TrainMetrics):
        metrics._storage._ep_rewards.window = deque([1.0, 2.0, 3.0])
        assert metrics.avg_reward() == 2.0

    def test_info(self, capsys, metrics: TrainMetrics):
        metrics._storage._ep_rewards.window = deque([1.0, 2.0, 3.0])
        metrics._storage._critic_losses.window = deque([0.1, 0.2, 0.3])
        metrics._storage._actor_losses.window = deque([0.4, 0.5, 0.6])

        metrics.info(current_ep=50)

        captured = capsys.readouterr()
        assert "Episode: 50/100" in captured.out
        assert "Avg Reward: 2.00" in captured.out
        assert "Critic Loss: 0.20" in captured.out
        assert "Actor Loss: 0.50" in captured.out

    def test_save(self, tmp_path, metrics: TrainMetrics):
        # Add some test data
        metrics._storage._ep_rewards.values = [1.0, 2.0]
        metrics._storage._ep_lengths.values = [10, 20]
        metrics._storage._critic_losses.values = [0.1, 0.2]
        metrics._storage._actor_losses.values = [0.3, 0.4]

        # Save metrics
        filepath = tmp_path / "test_metrics.pt"
        metrics.save(filepath)

        # Check if the file exists
        assert filepath.exists()

        # Load the saved data and verify
        data = torch.load(filepath)
        assert data["ep_rewards"] == [1.0, 2.0]
        assert data["ep_lengths"] == [10, 20]
        assert data["critic_losses"] == [0.1, 0.2]
        assert data["actor_losses"] == [0.3, 0.4]
