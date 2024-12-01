import pytest
import torch

from velora.agent.policy import EpsilonPolicy


class TestEpsilonPolicy:
    @pytest.fixture
    def policy(self) -> EpsilonPolicy:
        return EpsilonPolicy()

    @pytest.fixture
    def q_state(self) -> torch.FloatTensor:
        return torch.tensor([0.25, 0.25, 0.8, 0.25])

    @staticmethod
    def test_init_policy(policy: EpsilonPolicy):
        checks = [
            policy.epsilon == 1,
            policy.min_epsilon == 0.1,
            policy.decay_rate == 0.01,
            policy.device == torch.device("cpu"),
        ]
        assert all(checks)

    @staticmethod
    def test_custom_policy():
        policy = EpsilonPolicy(
            epsilon=0.5,
            min_epsilon=0.2,
            decay_rate=0.1,
            device=torch.device("cuda"),
        )

        checks = [
            policy.epsilon == 0.5,
            policy.min_epsilon == 0.2,
            policy.decay_rate == 0.1,
            policy.device == torch.device("cuda"),
        ]
        assert all(checks)

    @staticmethod
    def test_decay_linear(policy: EpsilonPolicy):
        before = policy.epsilon
        policy.decay_linear()

        checks = [
            policy.epsilon < before,
            policy.epsilon == before - 0.01,
        ]
        assert all(checks)

    @staticmethod
    def test_decay_exp(policy: EpsilonPolicy):
        before = policy.epsilon
        policy.decay_exp()

        checks = [
            policy.epsilon < before,
            policy.epsilon == before * 0.99,
        ]
        assert all(checks)

    @staticmethod
    def test_greedy_action(policy: EpsilonPolicy, q_state: torch.FloatTensor):
        policy.epsilon = policy.min_epsilon
        action = policy.greedy_action(q_state)
        assert action == 2

    @staticmethod
    def test_random_action(policy: EpsilonPolicy):
        action = policy.greedy_action(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        assert action in [0, 1, 2, 3]

    @staticmethod
    def test_as_dist(policy: EpsilonPolicy, q_state: torch.FloatTensor):
        result: torch.Tensor = policy.as_dist(q_state).probs
        assert (
            result.round(decimals=4) == torch.tensor([0.2113, 0.2113, 0.3662, 0.2113])
        ).all()

    @staticmethod
    def test_soft_dist(policy: EpsilonPolicy, q_state: torch.FloatTensor):
        result: torch.Tensor = policy.soft_dist(q_state).probs
        assert (result == torch.tensor([0.25, 0.25, 0.25, 0.25])).all()

    @staticmethod
    def test_soft_action(policy: EpsilonPolicy, q_state: torch.FloatTensor):
        torch.manual_seed(10)
        action = policy.soft_action(q_state)
        assert action == 2
