import pytest
from unittest.mock import Mock, patch
from functools import partial

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch

import torch

from velora.gym import (
    get_latest_env_names,
    wrap_gym_env,
    add_core_env_wrappers,
    get_action_bounds,
)


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

            mock_make.assert_called_once_with("CartPole-v1")
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


class TestGetActionBounds:
    @pytest.fixture
    def box_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)

    @pytest.fixture
    def discrete_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(5)

    def test_valid(self, box_space: gym.spaces.Box):
        low, high = get_action_bounds(box_space)
        assert low == -1.0
        assert high == 1.0

    def test_invalid(self, discrete_space: gym.spaces.Discrete):
        with pytest.raises(ValueError, match="action space is not supported"):
            get_action_bounds(discrete_space)


class TestGetLatestEnvNames:
    def test_valid_names(self):
        names = get_latest_env_names()
        assert ["Ant-v2", "Ant-v3", "Hopper-v3", "Hopper-v2"] not in names
