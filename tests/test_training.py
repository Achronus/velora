import json
from unittest.mock import MagicMock, patch
import pytest
from collections import deque

import numpy as np
from sqlmodel import select
import torch

from velora.metrics.models import Episode, Experiment, Step
from velora.models.base import RLAgent
from velora.models.config import (
    BufferConfig,
    ModelDetails,
    ModuleConfig,
    RLAgentConfig,
    TorchConfig,
)
from velora.models.ddpg import LiquidDDPG
from velora.state import RecordState
from velora.training.handler import TrainHandler
from velora.training.metrics import (
    StepStorage,
    MovingMetric,
    TrainMetrics,
)
from velora.utils.torch import summary


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

    def test_latest_property(self, metric: MovingMetric):
        metric.add(1.0)
        metric.add(2.0)

        assert metric.latest == 2.0


class TestTrainMetrics:
    @pytest.fixture
    def metrics(self, experiment) -> TrainMetrics:
        session, _ = experiment
        return TrainMetrics(session, window_size=10, n_episodes=100)

    @pytest.fixture
    def mock_config(self):
        ddpg = LiquidDDPG(4, 10, 1)

        return RLAgentConfig(
            agent="TestAgent",
            env="TestEnv",
            model_details=ModelDetails(
                type="actor-critic",
                state_dim=4,
                n_neurons=64,
                action_dim=2,
                target_networks=True,
                actor=ModuleConfig(
                    active_params=ddpg.actor.ncp.active_params,
                    total_params=ddpg.actor.ncp.total_params,
                    architecture=summary(ddpg.actor),
                ),
                critic=ModuleConfig(
                    active_params=ddpg.critic.ncp.active_params,
                    total_params=ddpg.critic.ncp.total_params,
                    architecture=summary(ddpg.critic),
                ),
            ),
            buffer=BufferConfig(
                type="ReplayBuffer",
                capacity=10000,
                state_dim=2,
                action_dim=1,
            ),
            torch=TorchConfig(device="cpu", optimizer="Adam", loss="MSELoss"),
        )

    def test_init(self, experiment):
        session, _ = experiment
        metrics = TrainMetrics(session, window_size=10, n_episodes=100)

        assert metrics.session == session
        assert metrics.window_size == 10
        assert metrics.n_episodes == 100
        assert metrics.experiment_id is None
        assert isinstance(metrics._ep_rewards, MovingMetric)
        assert metrics._ep_rewards.window_size == 10
        assert isinstance(metrics._current_losses, StepStorage)

    def test_start_experiment(self, metrics: TrainMetrics, mock_config):
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

    def test_add_step(self, metrics, experiment):
        """Test adding step metrics."""
        # Setup - start an experiment first
        session, experiment_id = experiment
        metrics.experiment_id = experiment_id

        mock_action = torch.tensor([0.5, -0.3])

        # Add a step
        metrics.add_step(
            ep_idx=5,
            step_idx=10,
            critic=0.8,
            actor=0.6,
            action=mock_action,
            action_threshold=0.3,
        )

        # Verify step was added using select
        statement = select(Step).where(
            Step.experiment_id == experiment_id,
            Step.episode_id == 5,
            Step.step_num == 10,
        )
        steps = session.exec(statement).all()

        assert len(steps) == 1
        step = steps[0]
        assert step.experiment_id == experiment_id
        assert step.episode_id == 5
        assert step.step_num == 10
        assert step.critic_loss == 0.8
        assert step.actor_loss == 0.6
        assert json.loads(step.action) == mock_action.tolist()

        # Verify losses were stored
        assert 0.8 in metrics._current_losses.critic_losses
        assert 0.6 in metrics._current_losses.actor_losses

    def test_add_step_without_experiment(self, metrics):
        """Test adding step metrics without first creating an experiment."""
        # Experiment ID is None (not set)
        with pytest.raises(RuntimeError) as excinfo:
            metrics.add_step(
                ep_idx=5,
                step_idx=10,
                critic=0.8,
                actor=0.6,
                action=torch.tensor([0.5]),
                action_threshold=0.3,
            )

        assert "An experiment must be created first" in str(excinfo.value)

    def test_add_episode(self, metrics, experiment):
        """Test adding episode metrics."""
        # Setup - start an experiment first
        session, experiment_id = experiment
        metrics.experiment_id = experiment_id

        # Add step metrics first (to have losses to average)
        metrics._current_losses.add({"critic_losses": 0.7, "actor_losses": 0.5})

        # Add an episode
        metrics.add_episode(ep_idx=5, reward=95.0, n_steps=200)

        # Verify episode was added using select
        statement = select(Episode).where(
            Episode.experiment_id == experiment_id, Episode.episode_num == 5
        )
        episodes = session.exec(statement).all()

        assert len(episodes) == 1
        episode = episodes[0]
        assert episode.experiment_id == experiment_id
        assert episode.episode_num == 5
        assert episode.reward == 95.0
        assert episode.length == 200
        assert episode.actor_loss == 0.5
        assert episode.critic_loss == 0.7

        # Verify current losses were emptied
        assert metrics._current_losses.critic_losses == []
        assert metrics._current_losses.actor_losses == []

        # Verify reward was added to moving metric
        assert 95.0 in metrics._ep_rewards.values

    def test_add_episode_without_experiment(self, metrics):
        """Test adding episode metrics without first creating an experiment."""
        # Experiment ID is None (not set)
        with pytest.raises(RuntimeError) as excinfo:
            metrics.add_episode(ep_idx=5, reward=95.0, n_steps=200)

        assert "An experiment must be created first" in str(excinfo.value)

    def test_reward_moving_avg(self, metrics):
        """Test calculating reward moving average."""
        # Add some rewards
        metrics._ep_rewards.add(80.0)
        metrics._ep_rewards.add(90.0)

        # Calculate average
        avg = metrics.reward_moving_avg()
        assert avg == 85.0  # Average of [80.0, 90.0]

    def test_reward_moving_std(self, metrics):
        """Test calculating reward moving standard deviation."""
        # Add some rewards
        metrics._ep_rewards.add(80.0)
        metrics._ep_rewards.add(90.0)

        # Calculate standard deviation
        std = metrics.reward_moving_std()
        assert std == 5.0  # std of [80.0, 90.0]

    def test_info(self, metrics, experiment):
        """Test info method that outputs to console."""
        # Setup
        session, experiment_id = experiment
        metrics.experiment_id = experiment_id

        # Create a test episode first
        episode = Episode(
            experiment_id=experiment_id,
            episode_num=50,
            reward=90.0,
            length=200,
            reward_moving_avg=85.0,
            reward_moving_std=5.0,
            actor_loss=0.6,
            critic_loss=0.7,
        )
        session.add(episode)
        session.commit()

        # Mock print
        with patch("builtins.print") as mock_print:
            # Execute
            metrics.info(current_ep=50)

            # Verify print was called
            mock_print.assert_called_once()

            # Check the print message format
            print_args = mock_print.call_args[0][0]
            assert "Episode: 50/100" in print_args
            assert "Max Reward:" in print_args
            assert "Avg Reward: 85.00" in print_args, mock_print.call_args
            assert "Critic Loss: 0.70" in print_args
            assert "Actor Loss: 0.60" in print_args


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
