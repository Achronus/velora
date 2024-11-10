from functools import partial
from pathlib import Path
import pytest

import gymnasium as gym
from gymnasium.wrappers import Autoreset, NormalizeObservation

from velora.config import Config
from velora.env import GymEnvHandler, wrap_gym_env


class TestWrapGymEnv:
    @pytest.fixture
    def base_env(self) -> gym.Env:
        return gym.make("CartPole-v1")

    @staticmethod
    def test_no_wrappers(base_env: gym.Env):
        wrapped_env = wrap_gym_env(base_env, [])
        assert wrapped_env == base_env

    @staticmethod
    def test_single_wrapper(base_env: gym.Env):
        wrapped_env = wrap_gym_env(base_env, [Autoreset])
        assert isinstance(wrapped_env, Autoreset)

    @staticmethod
    def test_partial_wrapper(base_env: gym.Env):
        wrapped_env = wrap_gym_env(
            base_env,
            [partial(NormalizeObservation, epsilon=1e-4)],
        )
        assert isinstance(wrapped_env, NormalizeObservation)

    @staticmethod
    def test_multiple_wrappers(base_env: gym.Env):
        wrappers = [
            Autoreset,
            partial(NormalizeObservation, epsilon=1e-4),
        ]

        wrapped_env = wrap_gym_env(base_env, wrappers)
        assert isinstance(wrapped_env, (Autoreset, NormalizeObservation))


class TestGymEnvHandler:
    @staticmethod
    def test_init(config_file: Path):
        handler = GymEnvHandler(config_filepath=config_file)

        checks = [
            isinstance(handler.config, Config),
            isinstance(handler.env, gym.Env),
            handler.env.spec.name == "CartPole",
            handler.wrappers == [],
        ]
        assert all(checks)

    @staticmethod
    def test_init_custom_env(config_file: Path):
        handler = GymEnvHandler(
            config_filepath=config_file,
            env=gym.make("MountainCar-v0"),
        )

        checks = [
            isinstance(handler.config, Config),
            isinstance(handler.env, gym.Env),
            handler.env.spec.name == "MountainCar",
            handler.wrappers == [],
        ]
        assert all(checks)

    @staticmethod
    def test_init_wrappers(config_file: Path):
        wrappers = [
            Autoreset,
            partial(NormalizeObservation, epsilon=1e-4),
        ]

        handler = GymEnvHandler(
            config_filepath=config_file,
            wrappers=wrappers,
        )

        checks = [
            isinstance(handler.config, Config),
            isinstance(handler.env, (gym.Env, Autoreset, NormalizeObservation)),
            handler.env.spec.name == "CartPole",
            handler.wrappers == wrappers,
        ]
        assert all(checks), handler.wrappers
