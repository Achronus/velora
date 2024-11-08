import pytest
import numpy as np

from velora import History, Trajectory


class TestHistory:
    @pytest.fixture
    def history(self) -> History:
        return History(
            items=[
                Trajectory(action=0, observation=np.zeros((2, 2)), reward=-1),
                Trajectory(action=1, observation=np.zeros((2, 2)), reward=-1),
                Trajectory(action=0, observation=np.zeros((2, 2)), reward=-1),
                Trajectory(action=1, observation=np.zeros((2, 2)), reward=10),
            ]
        )

    @staticmethod
    def test_actions(history: History):
        actions = history.actions()
        expected = [0, 1, 0, 1]

        checks = [
            isinstance(actions, list),
            actions == expected,
        ]
        assert all(checks)

    @staticmethod
    def test_observations(history: History):
        observations = history.observations()
        expected_observations = [
            np.array([[0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
        ]

        checks = [
            isinstance(observations, list),
            len(observations) == len(history.items),
        ]
        assert all(checks)

        for actual, expected in zip(observations, expected_observations):
            np.testing.assert_array_equal(actual, expected)

    @staticmethod
    def test_rewards(history: History):
        rewards = history.rewards()
        expected = [-1, -1, -1, 10]

        checks = [
            isinstance(rewards, list),
            rewards == expected,
        ]
        assert all(checks)

    @staticmethod
    def test_returns(history: History):
        """
        Test the returns calculation with γ=0.9.
        For rewards [-1, -1, -1, 10]:
        G₃ = 10
        G₂ = -1 + 0.9*10 = 8
        G₁ = -1 + 0.9*8 = 6.2
        G₀ = -1 + 0.9*6.2 = 4.58
        """
        G = history.returns(gamma=0.9)
        expected = [
            -1 + 0.9 * (-1 + 0.9 * (-1 + 0.9 * 10)),  # G₀
            -1 + 0.9 * (-1 + 0.9 * 10),  # G₁
            -1 + 0.9 * 10,  # G₂
            10,  # G₃
        ]

        assert len(G) == len(history.items)
        np.testing.assert_almost_equal(G, expected, decimal=5)

    @staticmethod
    def test_returns_no_discount(history: History):
        G = history.returns(gamma=1.0)
        expected = [7, 8, 9, 10]  # Sum of all future rewards
        np.testing.assert_almost_equal(G, expected)

    @staticmethod
    def test_returns_zero_discount(history: History):
        G = history.returns(gamma=0.0)
        expected = [-1, -1, -1, 10]  # Just immediate rewards
        np.testing.assert_almost_equal(G, expected)

    @staticmethod
    def test_empty_history():
        history = History(items=[])
        G = history.returns(gamma=0.9)
        assert len(G) == 0

    @staticmethod
    def test_single_trajectory():
        history = History(
            items=[
                Trajectory(action=0, observation=np.zeros((4, 4)), reward=5),
            ]
        )
        G = history.returns(gamma=0.9)
        checks = [len(G) == 1, G[0] == 5]
        assert all(checks)
