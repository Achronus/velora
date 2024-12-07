from pathlib import Path
from typing import Any, Type

import gymnasium as gym
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator
import torch

from velora.agent import Policy, ValueFunction
from velora.analytics import Analytics, NullAnalytics, WeightsAndBiases
from velora.metrics import Metrics
from velora.models import AgentModel, TorchAgentModel

from velora.config import Config, load_config
from velora.env import EnvHandler

from velora.utils import ignore_empty_dicts


class RLController(BaseModel):
    """
    Orchestrates the interactions between Agent, Environment, and Config.

    Args:
        config_filepath (pathlib.Path | str): a YAML config filepath
        env_handler (Type[velora.env.EnvHandler]): the type of environment handler to use
        agent_type (Type[velora.models.AgentModel | velora.models.TorchAgentModel]): the type of RL agent to use
        seed (int, optional): an random seed value for consistent experiments (default is 23)
        device (torch.device | str, optional): device to run computations on, such as `cpu`, `cuda` (Default is cpu)
        disable_logging (bool, optional): a flag to disable analytic logging (Default is False)
    """

    config_filepath: Path | str
    env_type: Type[EnvHandler]
    agent_type: Type[AgentModel | TorchAgentModel]
    seed: int = 23
    device: torch.device | str = torch.device("cpu")
    disable_logging: bool = False

    _config: Config = PrivateAttr(None)
    _agent: AgentModel = PrivateAttr(None)
    _env_handler: EnvHandler = PrivateAttr(None)
    _analytics: Analytics = PrivateAttr(None)
    _metrics: Metrics = PrivateAttr(default=Metrics())

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def env(self) -> gym.Env:
        """Returns the gym environment."""
        return self._env_handler.env

    @property
    def agent(self) -> AgentModel:
        """Returns the agent assigned to the controller."""
        return self._agent

    @property
    def config(self) -> Config:
        """Returns the Config model."""
        return self._config

    @property
    def policy(self) -> Policy:
        """Returns the agents policy."""
        return self._agent.policy

    @property
    def vf(self) -> ValueFunction:
        """Returns the agents value function."""
        return self._agent.vf

    @field_validator("device")
    def validate_device(cls, device: torch.device | str) -> torch.device:
        if isinstance(device, str):
            device = torch.device(device)

        return device

    def model_post_init(self, __context: Any) -> None:
        torch.manual_seed(self.seed)

        self._config = load_config(self.config_filepath)
        self._analytics = (
            NullAnalytics() if self.disable_logging else WeightsAndBiases()
        )

        self._env_handler = self.env_type(config=self._config.env)
        self._agent = self.agent_type(
            config=self._config,
            num_states=self.env.observation_space.n,
            num_actions=self.env.action_space.n,
            device=self.device,
        )

    def log_progress(self, ep_idx: int, log_count: int) -> None:
        """Displays helpful episode logs to the console during training."""
        if ep_idx % log_count == 0:
            self._metrics.update_percent(log_count)

            print(
                f"Episode {ep_idx}/{self.config.training.episodes} | ",
                end="",
            )
            print(self._metrics)

    def init_run(self, run_name: str) -> Analytics:
        """Creates a logging instance for analytics."""
        class_name = self.__class__.__name__
        self._analytics.init(
            project_name=f"{class_name}-{self._config.env.name}",
            run_name=run_name,
            config=ignore_empty_dicts(
                self._config.model_dump(
                    exclude=self.agent._config_exclusions,
                    exclude_none=True,
                )
            ),
            job_type=self.env.spec.name,
            tags=[self.env.spec.name, class_name],
        )
        return self._analytics

    def train(self, run_name: str, log_count: int = 100) -> None:
        """
        Trains the agent.

        Args:
            run_name (str): a unique name for the training run
            log_count (int, optional): the iteration number of episodes to log progress to the console. For example, when 100 always logs details every 100 episodes (default is 100)
        """
        run = self.init_run(run_name)

        for i_ep in range(1, self._config.training.episodes + 1):
            state, _ = self.env.reset()

            for _ in range(1, self._config.training.timesteps + 1):
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                loss = self.agent.step(state, next_state, action, reward)

                self._metrics.update_scores(reward, loss)

                state = next_state
                action = self.agent._next_action if self.agent._next_action else action

                if terminated or truncated:
                    self.agent.termination()
                    self._metrics.update_ep_counts(terminated)
                    break

            self.agent.finalize_episode()
            run.log(self._metrics.model_dump())
            self.log_progress(i_ep, log_count)

        run.finish()

    def predict(self) -> None:
        """Uses the trained agent to make a prediction."""
        pass

    def plot_vf(self) -> None:
        """Plots the value function."""
        pass
