import pytest
import torch

from velora.agent.storage import Rollouts
from velora.exc import RolloutsFullError


class TestRollouts:
    @pytest.fixture
    def history(self) -> Rollouts:
        h = Rollouts(size=4, obs_shape=(2, 2))
        h.add(obs=torch.zeros((2, 2)), action=0, reward=-1)
        h.add(obs=torch.zeros((2, 2)), action=1, reward=-1)
        h.add(obs=torch.zeros((2, 2)), action=0, reward=-1)
        h.add(obs=torch.zeros((2, 2)), action=1, reward=10)
        return h

    @staticmethod
    def test_add_full(history: Rollouts):
        with pytest.raises(RolloutsFullError):
            history.add(obs=torch.tensor((2, 2)), action=0, reward=-1)

    @staticmethod
    def test_returns(history: Rollouts):
        """
        Test the returns calculation with γ=0.9.
        For rewards [-1, -1, -1, 10]:
        G₃ = 10
        G₂ = -1 + 0.9*10 = 8
        G₁ = -1 + 0.9*8 = 6.2
        G₀ = -1 + 0.9*6.2 = 4.58
        """
        G = history.returns(gamma=0.9)
        expected = torch.FloatTensor(
            [
                -1 + 0.9 * (-1 + 0.9 * (-1 + 0.9 * 10)),  # G₀
                -1 + 0.9 * (-1 + 0.9 * 10),  # G₁
                -1 + 0.9 * 10,  # G₂
                10,  # G₃
            ]
        )

        assert len(G) == len(history)
        torch.testing.assert_close(G, expected)

    @staticmethod
    def test_returns_no_discount(history: Rollouts):
        G = history.returns(gamma=1.0)
        expected = torch.FloatTensor([7, 8, 9, 10])  # Sum of all future rewards
        torch.testing.assert_close(G, expected)

    @staticmethod
    def test_returns_zero_discount(history: Rollouts):
        G = history.returns(gamma=0.0)
        expected = torch.FloatTensor([-1, -1, -1, 10])  # Just immediate rewards
        torch.testing.assert_close(G, expected)

    @staticmethod
    def test_score(history: Rollouts):
        assert history.score() == 7

    @staticmethod
    def test_length(history: Rollouts):
        assert len(history) == 4

    @staticmethod
    def test_index_single(history: Rollouts):
        target = history[1]
        expected = (
            history._obs[1],
            history._actions[1],
            history._rewards[1],
        )

        checks = [
            target[0].eq(expected[0]).all(),
            target[1].eq(expected[1]).all(),
            target[2].eq(expected[2]).all(),
        ]
        assert all(checks)

    @staticmethod
    def test_slice(history: Rollouts):
        expected = Rollouts(size=2, obs_shape=(2, 2))
        expected.add(obs=torch.zeros((2, 2)), action=0, reward=-1)
        expected.add(obs=torch.zeros((2, 2)), action=1, reward=-1)

        target = history[:2]

        checks = [
            target.obs.eq(expected.obs).all(),
            target.actions.eq(expected.actions).all(),
            target.rewards.eq(expected.rewards).all(),
        ]
        assert all(checks)

    @staticmethod
    def test_repr(history: Rollouts):
        checks = [
            repr(history).startswith(f"Rollouts(size={history.size}, obs="),
            "actions=" in repr(history),
            "rewards=" in repr(history),
        ]
        assert all(checks)

    @staticmethod
    def test_str(history: Rollouts):
        assert str(history).startswith(
            "[(tensor([[0., 0.],\n        [0., 0.]]), 0, -1.0),\n"
        )
