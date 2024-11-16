import pytest
import torch

from velora import History, Trajectory, Episodes


class TestHistory:
    @pytest.fixture
    def history(self) -> History:
        h = History()
        h.extend(
            [
                Trajectory(action=0, observation=torch.tensor((2, 2)), reward=-1),
                Trajectory(action=1, observation=torch.tensor((2, 2)), reward=-1),
                Trajectory(action=0, observation=torch.tensor((2, 2)), reward=-1),
                Trajectory(action=1, observation=torch.tensor((2, 2)), reward=10),
            ]
        )
        return h

    @staticmethod
    def test_add(history: History):
        history.add(
            Trajectory(
                action=0,
                observation=torch.tensor((2, 2)),
                reward=-1,
            )
        )
        assert len(history) == 5

    @staticmethod
    def test_extend(history: History):
        history.extend(
            [
                Trajectory(
                    action=0,
                    observation=torch.tensor((2, 2)),
                    reward=-1,
                ),
                Trajectory(
                    action=0,
                    observation=torch.tensor((2, 2)),
                    reward=-1,
                ),
            ]
        )
        assert len(history) == 6

    @staticmethod
    def test_actions(history: History):
        actions = history.actions()
        expected = torch.LongTensor([0, 1, 0, 1])

        checks = [
            isinstance(actions, torch.LongTensor),
            actions.eq(expected).all(),
        ]
        assert all(checks), (actions, expected)

    @staticmethod
    def test_observations(history: History):
        observations = history.observations()
        expected_observations = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]])

        checks = [
            isinstance(observations, torch.Tensor),
            observations.shape == expected_observations.shape,
        ]
        assert all(checks), (observations.shape, expected_observations.shape)

    @staticmethod
    def test_rewards(history: History):
        rewards = history.rewards()
        expected = torch.LongTensor([-1, -1, -1, 10])

        checks = [
            isinstance(rewards, torch.Tensor),
            rewards.eq(expected).all(),
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
        expected = torch.FloatTensor(
            [
                -1 + 0.9 * (-1 + 0.9 * (-1 + 0.9 * 10)),  # G₀
                -1 + 0.9 * (-1 + 0.9 * 10),  # G₁
                -1 + 0.9 * 10,  # G₂
                10,  # G₃
            ]
        )

        assert len(G) == len(history._items)
        torch.testing.assert_close(G, expected)

    @staticmethod
    def test_returns_no_discount(history: History):
        G = history.returns(gamma=1.0)
        expected = torch.FloatTensor([7, 8, 9, 10])  # Sum of all future rewards
        torch.testing.assert_close(G, expected)

    @staticmethod
    def test_returns_zero_discount(history: History):
        G = history.returns(gamma=0.0)
        expected = torch.FloatTensor([-1, -1, -1, 10])  # Just immediate rewards
        torch.testing.assert_close(G, expected)

    @staticmethod
    def test_score(history: History):
        assert history.score() == 7

    @staticmethod
    def test_length(history: History):
        assert len(history) == 4

    @staticmethod
    def test_iter(history: History):
        items = list(history)
        assert len(items) == 4

    @staticmethod
    def test_indexing(history: History):
        assert history[1] == history._items[1]

    @staticmethod
    def test_empty_history():
        history = History(_items=[])
        G = history.returns(gamma=0.9)
        assert len(G) == 0

    @staticmethod
    def test_single_trajectory():
        history = History()

        history.add(
            Trajectory(
                action=0,
                observation=torch.tensor((4, 4)),
                reward=5,
            ),
        )
        G = history.returns(gamma=0.9)
        checks = [len(G) == 1, G[0] == 5]
        assert all(checks)


class TestEpisodes:
    @pytest.fixture
    def episodes(self) -> Episodes:
        h1 = History()
        h1.extend(
            [
                Trajectory(action=0, observation=torch.tensor((2, 2)), reward=-1),
                Trajectory(action=1, observation=torch.tensor((2, 2)), reward=-1),
                Trajectory(action=0, observation=torch.tensor((2, 2)), reward=-1),
                Trajectory(action=1, observation=torch.tensor((2, 2)), reward=10),
            ]
        )

        h2 = History()
        h2.extend(
            [
                Trajectory(action=0, observation=torch.tensor((2, 2)), reward=-1),
                Trajectory(action=1, observation=torch.tensor((2, 2)), reward=10),
            ]
        )

        e = Episodes()
        e.add(h1)
        e.add(h2)
        return e

    @staticmethod
    def test_add(episodes: Episodes):
        episodes.add(
            [
                Trajectory(action=0, observation=torch.tensor((2, 2)), reward=-1),
                Trajectory(action=1, observation=torch.tensor((2, 2)), reward=10),
            ]
        )
        assert len(episodes) == 3

    @staticmethod
    def test_scores(episodes: Episodes):
        result = episodes.scores()
        assert result.eq(torch.LongTensor([7, 9])).all()

    @staticmethod
    def test_observations(episodes: Episodes):
        observations = episodes.observations()
        expected_shape = (6, 2)
        assert observations.shape == expected_shape

    @staticmethod
    def test_actions(episodes: Episodes):
        actions = episodes.actions()
        expected = torch.tensor([0, 1, 0, 1, 0, 1])
        assert torch.equal(actions, expected)

    @staticmethod
    def test_length(episodes: Episodes):
        assert len(episodes) == 2

    @staticmethod
    def test_iter(episodes: Episodes):
        items = list(episodes)
        assert len(items) == 2

    @staticmethod
    def test_indexing(episodes: Episodes):
        expected = Episodes()
        expected.add(episodes._eps[-1])
        assert episodes[-1] == expected

    @staticmethod
    def test_slicing(episodes: Episodes):
        expected = Episodes()
        expected.add(episodes._eps[0])
        expected.add(episodes._eps[1])
        assert episodes[:2] == expected

    @staticmethod
    def test_add_objects_success(episodes: Episodes):
        result = episodes + episodes
        assert len(result) == 4

    @staticmethod
    def test_add_objects_fail(episodes: Episodes):
        with pytest.raises(NotImplementedError):
            episodes + History()
