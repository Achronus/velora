import pytest

import gymnasium as gym
import torch

from velora.agent.policy import EpsilonPolicy
from velora.agent.value import QTable
from velora.analytics.base import Analytics, NullAnalytics
from velora.config import Config
from velora.models.sarsa import SarsaBase, Sarsa, QLearning, ExpectedSarsa


class TestSarsaBase:
    @staticmethod
    def test_abstract(config: Config, env: gym.Env, device: torch.device):
        with pytest.raises(TypeError):
            SarsaBase(config=config, env=env, device=device)


class TestSarsa:
    @pytest.fixture
    def agent(self, config: Config, env: gym.Env, device: torch.device) -> Sarsa:
        return Sarsa(
            config=config,
            env=env,
            device=device,
            disable_logging=True,
        )

    @staticmethod
    def test_init(agent: Sarsa, env: gym.Env):
        checks = [
            isinstance(agent.Q, QTable),
            agent.Q.shape == (env.observation_space.n, env.action_space.n),
            isinstance(agent.policy, EpsilonPolicy),
        ]
        assert all(checks)

    @staticmethod
    def test_q_update(agent: Sarsa):
        q_value, q_next, reward = 0.5, 1.0, 0.1
        updated_q = agent.q_update(q_value, q_next, reward)

        assert updated_q == 0.005

    @staticmethod
    def test_init_run_with_tracking(config: Config, env: gym.Env, device: torch.device):
        agent = Sarsa(config=config, env=env, device=device)
        run = agent.init_run("test1")
        assert isinstance(run, Analytics)

    @staticmethod
    def test_init_run_without_tracking(agent: Sarsa):
        run = agent.init_run("test1")
        assert isinstance(run, NullAnalytics)

    @staticmethod
    def test_train(agent: Sarsa):
        Q = agent.train("test1")
        assert (
            Q.values[0].round(decimals=4)
            == torch.tensor([-0.01, 0.0, -0.0204, -0.0101])
        ).all(), Q.values[0]


class TestQLearning:
    @pytest.fixture
    def agent(self, config: Config, env: gym.Env, device: torch.device) -> QLearning:
        return QLearning(
            config=config,
            env=env,
            device=device,
            disable_logging=True,
        )

    @staticmethod
    def test_train(agent: QLearning):
        Q = agent.train("test1")
        assert (
            Q.values[0].round(decimals=4) == torch.tensor([-0.01, 0.0, -0.0201, -0.01])
        ).all(), Q.values[0]


class TestExpectedSarsa:
    @pytest.fixture
    def agent(
        self, config: Config, env: gym.Env, device: torch.device
    ) -> ExpectedSarsa:
        return ExpectedSarsa(
            config=config,
            env=env,
            device=device,
            disable_logging=True,
        )

    @staticmethod
    def test_train(agent: ExpectedSarsa):
        Q = agent.train("test1")
        assert (
            Q.values[0].round(decimals=4) == torch.tensor([-0.01, 0.0, -0.0202, -0.01])
        ).all(), Q.values[0]
