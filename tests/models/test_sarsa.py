import pytest

import gymnasium as gym
import torch

from velora.agent.policy import EpsilonPolicy
from velora.agent.value import QTable
from velora.config import Config
from velora.models.sarsa import SarsaBase, Sarsa, QLearning, ExpectedSarsa


class TestSarsaBase:
    @staticmethod
    def test_abstract(config: Config, env: gym.Env, device: torch.device):
        with pytest.raises(TypeError):
            SarsaBase(
                config=config,
                num_states=env.observation_space.n,
                num_actions=env.action_space.n,
                device=device,
            )


class TestSarsa:
    @pytest.fixture
    def agent(self, config: Config, env: gym.Env, device: torch.device) -> Sarsa:
        return Sarsa(
            config=config,
            num_states=env.observation_space.n,
            num_actions=env.action_space.n,
            device=device,
        )

    @staticmethod
    def test_init(agent: Sarsa, env: gym.Env):
        checks = [
            isinstance(agent.vf, QTable),
            agent.vf.shape == (env.observation_space.n, env.action_space.n),
            isinstance(agent.policy, EpsilonPolicy),
        ]
        assert all(checks)

    @staticmethod
    def test_q_update(agent: Sarsa):
        q_value, q_next, reward = 0.5, 1.0, 0.1
        td_error = agent.td_error(q_value, q_next, reward)

        assert agent.q_update(td_error) == 0.005


class TestQLearning:
    @staticmethod
    def test_init(config: Config, env: gym.Env, device: torch.device) -> QLearning:
        agent = QLearning(
            config=config,
            num_states=env.observation_space.n,
            num_actions=env.action_space.n,
            device=device,
        )
        assert isinstance(agent, QLearning)


class TestExpectedSarsa:
    @staticmethod
    def test_init(config: Config, env: gym.Env, device: torch.device) -> ExpectedSarsa:
        agent = ExpectedSarsa(
            config=config,
            num_states=env.observation_space.n,
            num_actions=env.action_space.n,
            device=device,
        )
        assert isinstance(agent, ExpectedSarsa)
