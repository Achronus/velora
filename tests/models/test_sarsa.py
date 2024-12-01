import pytest

import gymnasium as gym
import torch
from wandb.sdk.wandb_run import Run

from velora.agent.policy import EpsilonPolicy
from velora.agent.value import QTable
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
        return Sarsa(config=config, env=env, device=device)

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
    def test_init_run(agent: Sarsa):
        run = agent.init_run("test1")
        assert isinstance(run, Run)
