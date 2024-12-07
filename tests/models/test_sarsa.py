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
        updated_q = agent.q_update(q_value, q_next, reward)

        assert updated_q == 0.005

    @staticmethod
    @pytest.mark.parametrize(
        "ep_idx, log_count, expected",
        [
            (1, 1, "Episode 1/1"),
            (7, 5, ""),  # No print when not divisible by log_count
        ],
    )
    def test_log_progress(
        capfd, ep_idx: int, log_count: int, expected: str, agent: Sarsa
    ):
        agent.log_progress(ep_idx, log_count)

        result = capfd.readouterr()
        assert result.out.strip() == expected


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
