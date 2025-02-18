import pytest
from unittest.mock import Mock, patch
from functools import partial

import gymnasium as gym
from gymnasium import spaces

from velora.gym import get_latest_env_names, wrap_gym_env


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


class TestGetLatestEnvNames:
    def test_valid_names(self):
        names = get_latest_env_names()
        assert ["Ant-v2", "Ant-v3", "Hopper-v3", "Hopper-v2"] not in names
