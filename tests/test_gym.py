import pytest
from unittest.mock import Mock, patch
from functools import partial

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch

import torch

from velora.gym import wrap_gym_env, add_core_env_wrappers, EnvSearch, SearchHelper


class TestWrapGymEnv:
    @pytest.fixture
    def mock_env(self):
        env = Mock(spec=gym.Env)
        env.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        env.action_space = spaces.Discrete(2)
        return env

    def test_wrap_env_with_string(self):
        with patch("gymnasium.make") as mock_make:
            mock_env = Mock(spec=gym.Env)
            mock_make.return_value = mock_env

            wrapped_env = wrap_gym_env("CartPole-v1", [])

            mock_make.assert_called_once_with("CartPole-v1", render_mode="rgb_array")
            assert wrapped_env == mock_env

    def test_wrap_env_with_wrappers(self, mock_env):
        class TestWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)

        wrapped_env = wrap_gym_env(mock_env, [TestWrapper])
        assert isinstance(wrapped_env, TestWrapper)

    def test_wrap_env_with_partial_wrappers(self, mock_env):
        class TestWrapper(gym.Wrapper):
            def __init__(self, env, param):
                super().__init__(env)
                self.param = param

        wrapper = partial(TestWrapper, param=42)
        wrapped_env = wrap_gym_env(mock_env, [wrapper])

        assert isinstance(wrapped_env, TestWrapper)
        assert wrapped_env.param == 42

    def test_multiple_wrappers(self, mock_env):
        class WrapperA(gym.Wrapper):
            pass

        class WrapperB(gym.Wrapper):
            pass

        wrapped_env = wrap_gym_env(mock_env, [WrapperA, WrapperB])

        assert isinstance(wrapped_env, WrapperB)
        assert isinstance(wrapped_env.env, WrapperA)


class TestAddCoreEnvWrappers:
    @pytest.fixture
    def mock_env(self) -> gym.Env:
        return gym.make("CartPole-v1")

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cpu")

    def test_applies_wrappers(self, mock_env: gym.Env, device: torch.device):
        wrapped_env = add_core_env_wrappers(mock_env, device)

        assert isinstance(wrapped_env, NumpyToTorch)
        assert isinstance(wrapped_env.env, RecordEpisodeStatistics)

    def test_duplicate_wrappers(self, mock_env: gym.Env, device: torch.device):
        wrapped_once = add_core_env_wrappers(mock_env, device)
        wrapped_twice = add_core_env_wrappers(wrapped_once, device)

        assert isinstance(wrapped_twice, NumpyToTorch)
        assert isinstance(wrapped_twice.env, RecordEpisodeStatistics)
        assert wrapped_once is wrapped_twice  # No redundant wrapping

    def test_invalid_environment(self, device: torch.device):
        envs = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")

        with pytest.raises(ValueError):
            add_core_env_wrappers(envs, device)


class TestSearchHelper:
    @pytest.fixture
    def mock_gym(self):
        with patch("gymnasium.make") as mock_make:
            yield mock_make

    def test_get_env_type_discrete(self, mock_gym):
        mock_env = Mock()
        mock_env.action_space = spaces.Discrete(4)
        mock_gym.return_value = mock_env

        result = SearchHelper.get_env_type("TestEnv-v0")
        assert result == "discrete"
        mock_env.close.assert_called_once()

    def test_get_env_type_continuous(self, mock_gym):
        mock_env = Mock()
        mock_env.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        mock_gym.return_value = mock_env

        result = SearchHelper.get_env_type("TestEnv-v0")
        assert result == "continuous"
        mock_env.close.assert_called_once()

    def test_get_env_type_other(self, mock_gym):
        mock_env = Mock()
        mock_env.action_space = Mock()  # Neither Discrete nor Box
        mock_gym.return_value = mock_env

        result = SearchHelper.get_env_type("TestEnv-v0")
        assert result is None
        mock_env.close.assert_called_once()

    def test_get_env_type_error(self, mock_gym):
        mock_gym.side_effect = Exception("Test error")

        with pytest.raises(Exception):
            SearchHelper.get_env_type("InvalidEnv-v0")

    def test_get_latest_env_names(self):
        mock_registry = {
            "CartPole-v0": None,
            "CartPole-v1": None,
            "Pendulum-v1": None,
            "GymV26Environment": None,
            "jax/Pendulum-v0": None,
        }

        with patch("gymnasium.envs.registry", mock_registry):
            result = SearchHelper.get_latest_env_names()

            assert "CartPole-v1" in result  # Latest version should be included
            assert "CartPole-v0" not in result  # Older version should be filtered
            assert "Pendulum-v1" in result
            assert "GymV26Environment" not in result
            assert "jax/Pendulum-v0" not in result


class TestEnvSearch:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Clear cache before each test
        EnvSearch._result_cache = []

    @pytest.fixture
    def mock_helper(self):
        with (
            patch.object(SearchHelper, "get_latest_env_names") as mock_names,
            patch.object(SearchHelper, "get_env_type") as mock_type,
        ):
            yield mock_names, mock_type

    def test_find_matching_environments(self, mock_helper):
        mock_names, mock_type = mock_helper
        mock_names.return_value = ["CartPole-v1", "Pendulum-v1"]
        mock_type.side_effect = ["discrete", "continuous"]

        # First search builds cache
        results = EnvSearch.find("Cart")
        assert len(results) == 1
        assert results[0].name == "CartPole-v1"
        assert results[0].type == "discrete"

        # Second search uses cache
        results = EnvSearch.find("Pend")
        assert len(results) == 1
        assert results[0].name == "Pendulum-v1"
        assert results[0].type == "continuous"

        # Verify cache is only built once
        mock_names.assert_called_once()

    def test_discrete_environments(self, mock_helper):
        mock_names, mock_type = mock_helper
        mock_names.return_value = ["CartPole-v1", "Pendulum-v1"]
        mock_type.side_effect = ["discrete", "continuous"]

        results = EnvSearch.discrete()
        assert len(results) == 1
        assert results[0].name == "CartPole-v1"
        assert results[0].type == "discrete"

    def test_continuous_environments(self, mock_helper):
        mock_names, mock_type = mock_helper
        mock_names.return_value = ["CartPole-v1", "Pendulum-v1"]
        mock_type.side_effect = ["discrete", "continuous"]

        results = EnvSearch.continuous()
        assert len(results) == 1
        assert results[0].name == "Pendulum-v1"
        assert results[0].type == "continuous"

    def test_empty_search_results(self, mock_helper):
        mock_names, mock_type = mock_helper
        mock_names.return_value = ["CartPole-v1", "Pendulum-v1"]
        mock_type.side_effect = ["discrete", "continuous"]

        results = EnvSearch.find("NonexistentEnv")
        assert len(results) == 0

    def test_cache_reuse(self, mock_helper):
        mock_names, mock_type = mock_helper
        mock_names.return_value = ["CartPole-v1"]
        mock_type.return_value = "discrete"

        # First call should build cache
        EnvSearch.find("Cart")
        initial_call_count = mock_names.call_count

        # Subsequent calls should reuse cache
        EnvSearch.find("Cart")
        EnvSearch.discrete()
        EnvSearch.continuous()

        assert mock_names.call_count == initial_call_count
