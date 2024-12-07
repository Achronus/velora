from functools import partial
from typing import Callable
import pytest
from pathlib import Path

from velora.agent.policy import EpsilonPolicy
from velora.analytics.base import NullAnalytics
from velora.analytics.wandb import WeightsAndBiases
from velora.config import Config
from velora.controller import RLController
from velora.env.gym import GymEnvHandler
from velora.models import Sarsa, QLearning, ExpectedSarsa

import torch


PartialGym = Callable[[str], RLController]


class TestRLController:
    @pytest.fixture
    def sarsa_gym(self, config_file: Path) -> RLController:
        return RLController(
            config_filepath=config_file,
            env_type=GymEnvHandler,
            agent_type=Sarsa,
            disable_logging=True,
        )

    @pytest.fixture
    def gym_controller(self, config_file: Path) -> PartialGym:
        return partial(
            RLController,
            config_filepath=config_file,
            env_type=GymEnvHandler,
            disable_logging=True,
        )

    @staticmethod
    def test_init_run_with_tracking(config_file: Path):
        controller = RLController(
            config_filepath=config_file,
            env_type=GymEnvHandler,
            agent_type=Sarsa,
        )
        controller.init_run("test1")
        assert isinstance(controller._analytics, WeightsAndBiases)

    @staticmethod
    def test_init_run_without_tracking(config_file: Path):
        controller = RLController(
            config_filepath=config_file,
            env_type=GymEnvHandler,
            agent_type=Sarsa,
            disable_logging=True,
        )
        controller.init_run("test1")
        assert isinstance(controller._analytics, NullAnalytics), type(
            controller._analytics
        )

    @staticmethod
    def test_sarsa_train(gym_controller: PartialGym):
        agent: RLController = gym_controller(agent_type=Sarsa)
        vf_init = agent.vf.values

        agent.train("test1")
        assert (agent.vf.values >= vf_init).all()

    @staticmethod
    def test_qlearning_train(gym_controller: PartialGym):
        agent: RLController = gym_controller(agent_type=QLearning)
        vf_init = agent.vf.values

        agent.train("test1")
        assert (agent.vf.values >= vf_init).all()

    @staticmethod
    def test_expected_sarsa_train(gym_controller: PartialGym):
        agent: RLController = gym_controller(agent_type=ExpectedSarsa)
        vf_init = agent.vf.values

        agent.train("test1")
        assert (agent.vf.values >= vf_init).all()

    @staticmethod
    def test_config_property(sarsa_gym: RLController):
        assert isinstance(sarsa_gym.config, Config)

    @staticmethod
    def test_policy_property(sarsa_gym: RLController):
        assert isinstance(sarsa_gym.policy, EpsilonPolicy)

    @staticmethod
    def test_device_validation_str(config_file: Path):
        agent = RLController(
            config_filepath=config_file,
            env_type=GymEnvHandler,
            agent_type=Sarsa,
            disable_logging=True,
            device="cpu",
        )

        assert agent.device == torch.device("cpu")
