from unittest.mock import MagicMock, patch
import pytest

import torch

from velora.metrics.db import get_current_episode
from velora.metrics.models import Experiment
from velora.models.base import RLAgent

from velora.models.config import RLAgentConfig
from velora.models.ddpg import LiquidDDPG
from velora.state import RecordState
from velora.training.handler import TrainHandler
from velora.training.metrics import (
    StepStorage,
    MovingMetric,
    TrainMetrics,
)


class TestStepStorage:
    @pytest.fixture
    def storage(self) -> StepStorage:
        return StepStorage(capacity=10, device=torch.device("cpu"))

    def test_init(self, storage: StepStorage):
        assert storage.capacity == 10
        assert storage.position == 0
        assert storage.size == 0
        assert storage.critic_losses.shape == (10,)
        assert storage.actor_losses.shape == (10,)

    def test_critic_avg(self, storage: StepStorage):
        # Fill the storage with some values
        storage.critic_losses[0:3] = torch.tensor([1.0, 2.0, 3.0])

        # Test calculating average for a subset of entries
        avg = storage.critic_avg(3)
        assert avg.item() == 2.0  # Average of [1.0, 2.0, 3.0]

    def test_actor_avg(self, storage: StepStorage):
        # Fill the storage with some values
        storage.actor_losses[0:3] = torch.tensor([1.0, 2.0, 3.0])

        # Test calculating average for a subset of entries
        avg = storage.actor_avg(3)
        assert avg.item() == 2.0  # Average of [1.0, 2.0, 3.0]

    def test_add(self, storage: StepStorage):
        critic_loss = torch.tensor(2.5)
        actor_loss = torch.tensor(3.5)

        storage.add(critic_loss, actor_loss)

        assert storage.critic_losses[0].item() == 2.5
        assert storage.actor_losses[0].item() == 3.5
        assert storage.position == 1
        assert storage.size == 1

    def test_add_multiple(self, storage: StepStorage):
        for i in range(12):  # More than capacity to test wrapping
            storage.add(torch.tensor(i), torch.tensor(i + 10))

        # Check that position wrapped around
        assert storage.position == 2
        assert storage.size == 10  # Capped at capacity

        # Check the most recent values (should be the wrapped ones)
        assert storage.critic_losses[0].item() == 10
        assert storage.critic_losses[1].item() == 11

    def test_empty(self, storage: StepStorage):
        # Add some data
        storage.critic_losses[0:3] = torch.tensor([1.0, 2.0, 3.0])
        storage.actor_losses[0:3] = torch.tensor([4.0, 5.0, 6.0])
        storage.position = 3
        storage.size = 3

        # Empty the storage
        storage.empty()

        # Check everything is reset
        assert storage.position == 0
        assert storage.size == 0
        assert torch.all(storage.critic_losses == 0)
        assert torch.all(storage.actor_losses == 0)


class TestMovingMetric:
    @pytest.fixture
    def metric(self) -> MovingMetric:
        return MovingMetric(window_size=5, device=torch.device("cpu"))

    def test_init(self, metric: MovingMetric):
        assert metric.window_size == 5
        assert metric.window.shape == (5,)
        assert metric.position == 0
        assert metric.size == 0

    def test_latest(self, metric: MovingMetric):
        # Add some values
        metric.add(torch.tensor(1.0))
        metric.add(torch.tensor(2.0))

        # Check the latest value
        assert metric.latest.item() == 2.0

    def test_latest_wrapped(self, metric: MovingMetric):
        # Fill up and wrap around
        for i in range(6):
            metric.add(torch.tensor(float(i)))

        # The latest value should be 5
        assert metric.latest.item() == 5.0

    def test_add(self, metric: MovingMetric):
        metric.add(torch.tensor(3.0))

        assert metric.window[0].item() == 3.0
        assert metric.position == 1
        assert metric.size == 1

    def test_add_exceeds_window_size(self, metric: MovingMetric):
        # Add more items than the window size
        for i in range(7):
            metric.add(torch.tensor(float(i)))

        # Position should have wrapped around
        assert metric.position == 2
        assert metric.size == 5  # Capped at window_size

        # The window should now contain the 5 most recent values [2,3,4,5,6]
        # Due to wrapping, they're not in sequential order in memory
        expected_values = {2.0, 3.0, 4.0, 5.0, 6.0}
        actual_values = {v.item() for v in metric.window}
        assert expected_values == actual_values

    def test_mean(self, metric: MovingMetric):
        # Add some values
        for i in range(3):
            metric.add(torch.tensor(float(i + 1)))  # [1, 2, 3]

        # Calculate mean
        assert torch.isclose(metric.mean(), torch.tensor(1.2))

    def test_std(self, metric: MovingMetric):
        # Add some values
        for i in range(3):
            metric.add(torch.tensor(float(i + 1)))  # [1, 2, 3]

        # Calculate standard deviation
        actual_std = metric.std()
        assert torch.isclose(actual_std, torch.tensor(1.3), rtol=0.01)

    def test_max(self, metric: MovingMetric):
        # Add some values
        for i in range(3):
            metric.add(torch.tensor(float(i + 1)))  # [1, 2, 3]

        # Calculate max
        assert metric.max().item() == 3.0

    def test_len(self, metric: MovingMetric):
        assert len(metric) == 0

        # Add some values
        for i in range(3):
            metric.add(torch.tensor(float(i)))

        assert len(metric) == 3


class TestTrainMetrics:
    @pytest.fixture
    def metrics(self, experiment) -> TrainMetrics:
        session, _ = experiment
        return TrainMetrics(
            session,
            window_size=10,
            n_episodes=100,
            max_steps=10,
            device=torch.device("cpu"),
        )

    @pytest.fixture
    def mock_config(self) -> RLAgentConfig:
        ddpg = LiquidDDPG(4, 10, 1)

        config = ddpg.config
        config = config.update(
            "TestEnv",
            dict(
                batch_size=32,
                n_episodes=100,
                max_steps=200,
                window_size=10,
                gamma=0.99,
                tau=0.005,
                noise_scale=0.3,
            ),
        )

        return config

    def test_init(self, experiment):
        session, _ = experiment
        metrics = TrainMetrics(
            session,
            window_size=10,
            n_episodes=100,
            max_steps=200,
            device=torch.device("cpu"),
        )

        assert metrics.session == session
        assert metrics.window_size == 10
        assert metrics.n_episodes == 100
        assert metrics.max_steps == 200
        assert metrics.experiment_id is None
        assert isinstance(metrics._ep_rewards, MovingMetric)
        assert metrics._ep_rewards.window_size == 10
        assert isinstance(metrics._current_losses, StepStorage)
        assert metrics._current_losses.capacity == 200

    def test_start_experiment(self, metrics: TrainMetrics, mock_config: RLAgentConfig):
        """Test starting an experiment."""
        # Start the experiment
        metrics.start_experiment(mock_config)

        # Verify experiment was created
        assert metrics.experiment_id is not None

        # Verify the experiment exists in the database
        experiment = metrics.session.get(Experiment, metrics.experiment_id)
        assert experiment is not None
        assert experiment.agent == mock_config.agent
        assert experiment.env == mock_config.env

    def test_add_step(self, metrics: TrainMetrics):
        """Test adding step metrics."""
        # Fill in test values
        critic_loss = torch.tensor(0.8)
        actor_loss = torch.tensor(0.6)

        # Add a step
        metrics.add_step(critic=critic_loss, actor=actor_loss)

        # Verify losses were stored
        assert torch.isclose(
            metrics._current_losses.critic_losses[0], torch.tensor(0.8)
        )
        assert torch.isclose(metrics._current_losses.actor_losses[0], torch.tensor(0.6))
        assert metrics._current_losses.position == 1
        assert metrics._current_losses.size == 1

    def test_add_episode(self, metrics: TrainMetrics, experiment):
        """Test adding episode metrics."""
        # Setup - start an experiment first
        session, experiment_id = experiment
        metrics.experiment_id = experiment_id

        # Add step metrics first (to have losses to average)
        metrics._current_losses.add(torch.tensor(0.7), torch.tensor(0.5))

        # Add an episode
        ep_idx = 5
        reward = torch.tensor(95.0)
        ep_length = torch.tensor(200)
        metrics.add_episode(ep_idx=ep_idx, reward=reward, ep_length=ep_length)

        # Query the database for the episode
        episodes = get_current_episode(session, experiment_id, ep_idx)

        results = []
        for ep in episodes:
            results.append(ep)

        assert len(results) == 1
        episode = results[0]
        assert episode.experiment_id == experiment_id
        assert episode.episode_num == ep_idx
        assert episode.reward == reward.item()
        assert episode.length == ep_length.item()
        assert torch.isclose(torch.tensor(episode.actor_loss), torch.tensor(0.05))
        assert torch.isclose(torch.tensor(episode.critic_loss), torch.tensor(0.07))

        # Verify reward was added to moving metric
        assert torch.isclose(
            torch.tensor(metrics._ep_rewards.window[0]),
            torch.tensor(95.0),
        )

        # Verify current losses were emptied
        assert metrics._current_losses.position == 0
        assert metrics._current_losses.size == 0

    def test_add_episode_without_experiment(self, metrics: TrainMetrics):
        """Test adding episode metrics without first creating an experiment."""
        # Experiment ID is None (not set)
        with pytest.raises(RuntimeError) as excinfo:
            metrics.add_episode(
                ep_idx=5, reward=torch.tensor(95.0), ep_length=torch.tensor(200)
            )

        assert "An experiment must be created first" in str(excinfo.value)

    def test_reward_moving_avg(self, metrics: TrainMetrics):
        """Test calculating reward moving average."""
        # Add some rewards
        metrics._ep_rewards.add(torch.tensor(80.0))
        metrics._ep_rewards.add(torch.tensor(90.0))

        # Calculate average
        avg = metrics.reward_moving_avg()
        assert torch.isclose(torch.tensor(avg), torch.tensor(17.0)), avg

    def test_reward_moving_std(self, metrics: TrainMetrics):
        """Test calculating reward moving standard deviation."""
        # Add some rewards
        metrics._ep_rewards.add(torch.tensor(80.0))
        metrics._ep_rewards.add(torch.tensor(90.0))

        # Calculate standard deviation
        std = metrics.reward_moving_std()
        assert torch.isclose(torch.tensor(std), torch.tensor(35.91656)), std

    def test_reward_moving_max(self, metrics: TrainMetrics):
        """Test calculating reward moving max."""
        # Add some rewards
        metrics._ep_rewards.add(torch.tensor(80.0))
        metrics._ep_rewards.add(torch.tensor(90.0))

        # Calculate max
        max_reward = metrics.reward_moving_max()
        assert torch.isclose(torch.tensor(max_reward), torch.tensor(90.0))

    def test_info(self, metrics: TrainMetrics):
        """Test info method that outputs to console."""
        # Setup
        metrics._ep_lengths = torch.zeros(
            (100,), dtype=torch.int, device=torch.device("cpu")
        )
        metrics._ep_lengths[49] = 200  # Set episode 50's length
        metrics._ep_rewards.add(torch.tensor(85.0))
        metrics._critic_loss = torch.tensor(0.7)
        metrics._actor_loss = torch.tensor(0.6)

        # Mock print
        with patch("builtins.print") as mock_print:
            # Execute
            metrics.info(current_ep=50)

            # Verify print was called
            mock_print.assert_called_once()

            # Check the print message format (don't check exact format as it might change)
            print_args = mock_print.call_args[0][0]
            assert "Episode" in print_args
            assert "Length:" in print_args
            assert "Reward Avg:" in print_args
            assert "Reward Max:" in print_args
            assert "Critic Loss:" in print_args
            assert "Actor Loss:" in print_args


class TestTrainHandler:
    def test_episode_method_updates_state(self):
        # Create mock objects
        mock_state = MagicMock()
        mock_callback = MagicMock()

        # Set up the TrainHandler instance with mocks
        handler = TrainHandler(
            agent=MagicMock(),
            env=MagicMock(),
            n_episodes=100,
            max_steps=200,
            window_size=10,
            callbacks=[mock_callback],
        )

        # Assign the mock state to the handler
        handler.state = mock_state

        # Define test values
        current_ep = 5
        ep_reward = 120.5

        # Call the method being tested
        handler.episode(current_ep, ep_reward)

        # Verify the state.update was called with correct parameters
        mock_state.update.assert_called_once_with(
            status="episode",
            current_ep=current_ep,
            ep_reward=ep_reward,
        )

        # Verify that the callback was called once with the state
        mock_callback.assert_called_once_with(mock_state)


class TestTrainHandlerRecordLastEpisode:
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock(spec=RLAgent)
        agent.device = "cpu"
        return agent

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.spec.id = "TestEnv-v0"
        env.spec.name = "TestEnv"
        return env

    @pytest.fixture
    def train_handler(self, mock_agent, mock_env, experiment):
        handler = TrainHandler(
            agent=mock_agent,
            env=mock_env,
            n_episodes=100,
            max_steps=10,
            window_size=10,
            callbacks=None,
        )
        handler.session = experiment[0]
        handler._metrics = MagicMock()

        return handler

    @patch("velora.training.handler.record_last_episode")
    def test_record_last_episode_with_record_state(
        self, mock_record, train_handler, mock_agent, mock_env, tmp_path
    ):
        # Set up record_state in the handler's state
        test_dir = tmp_path / "test_videos"
        test_dir.mkdir()

        train_handler.state = MagicMock()
        train_handler.state.agent = mock_agent
        train_handler.state.env = mock_env
        train_handler.state.record_state = RecordState(
            dirpath=test_dir, method="episode", episode_trigger=lambda x: x % 10 == 0
        )

        # Call the record_last_episode method
        train_handler.record_last_episode()

        # Verify the record_last_episode function was called with correct parameters
        mock_record.assert_called_once_with(
            mock_agent, mock_env.spec.id, test_dir.parent.name
        )

    @patch("velora.training.handler.record_last_episode")
    def test_record_last_episode_without_record_state(self, mock_record, train_handler):
        # Set up state without record_state
        train_handler.state = MagicMock()
        train_handler.state.record_state = None

        # Call the record_last_episode method
        train_handler.record_last_episode()

        # Verify the record_last_episode function was not called
        mock_record.assert_not_called()
