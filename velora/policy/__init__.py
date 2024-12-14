from pathlib import Path
import time
from typing import Any, Type

import gymnasium as gym
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

import torch
import torch.optim as optim
from torch.nn.modules.loss import _Loss

import numpy as np

from velora.agent.storage import Rollouts
from velora.analytics import Analytics, NullAnalytics, WeightsAndBiases
from velora.env.gym import GymEnv

from velora.metrics import Metrics
from velora.policy.base import AgentModel, FeatureExtractor
from velora.policy.ppo import PPO, PPOInputs

from velora.config import Config, load_config
from velora.utils import ignore_empty_dicts


__all__ = [
    "AgentModel",
    "FeatureExtractor",
]


class RLAgent(BaseModel):
    """
    Creates a Reinforcement Learning agent.

    Args:
        config_filepath (pathlib.Path | str): a YAML config filepath
        agent (Type[velora.agent.AgentModel]): the type of PyTorch agent model
        optimizer_type (Type[torch.optim.Optimizer]): the type of PyTorch optimizer
        loss_type (Type[torch.nn.Loss]): the type of PyTorch loss criterion
        device (torch.device | str, optional): device to run computations on, such as `cpu`, `cuda` (Default is cpu)
        disable_logging (bool, optional): a flag to disable analytic logging. If True creates a [Weights and Bias](https://wandb.ai/) instance (Default is False)
    """

    config_filepath: Path | str
    agent: Type[AgentModel]
    optimizer_type: Type[optim.Optimizer]
    loss_type: Type[_Loss]
    device: torch.device | str = torch.device("cpu")
    disable_logging: bool = False

    _config: Config = PrivateAttr(None)
    _env_handler: GymEnv = PrivateAttr(None)
    _storage: Rollouts = PrivateAttr(None)

    _model: AgentModel = PrivateAttr(None)
    _optimizer: optim.Optimizer = PrivateAttr(None)
    _loss: _Loss = PrivateAttr(None)

    _analytics: Analytics = PrivateAttr(None)
    _metrics: Metrics = PrivateAttr(default=Metrics())

    _batch_size: int = PrivateAttr(...)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def env(self) -> gym.vector.SyncVectorEnv:
        """Returns the gym environment."""
        return self._env_handler.env

    @property
    def model(self) -> AgentModel:
        """Returns the agent model."""
        return self._model

    @property
    def config(self) -> Config:
        """Returns the Config model."""
        return self._config

    @field_validator("device")
    def validate_device(cls, device: torch.device | str) -> torch.device:
        if isinstance(device, str):
            device = torch.device(device)

        return device

    def model_post_init(self, __context: Any) -> None:
        self.set_seed(self.config.run.seed)

        self._config = load_config(self.config_filepath)
        self._analytics = (
            NullAnalytics() if self.disable_logging else WeightsAndBiases()
        )
        self._env_handler = GymEnv(config=self._config)

        self._model = self.agent(
            continuous=self._env_handler._continuous_actions,
            **self._agent_args(),
        ).to(self.device)
        self._optimizer = self.optimizer_type(
            self._model.parameters(), **self.config.optimizer
        )
        self._loss = self.loss_type(**self.config.loss)

        self._storage = Rollouts(
            size=self.config.agent.buffer_size,
            n_envs=self.config.env.n_envs,
            n_actions=self._env_handler.n_actions,
            obs_shape=self.env.single_observation_space.shape,
            device=self.device,
        )
        self._batch_size = int(self._storage.n_envs * self._storage.size)

    def set_seed(self, seed: int) -> None:
        """Sets the seed to ensure reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _agent_args(self) -> dict[str, Any]:
        """Sets the agents unique arguments and returns them as a dictionary."""
        if isinstance(self.agent, PPO):
            return PPOInputs(
                in_channels=self._env_handler.n_states,
                n_actions=self._env_handler.n_actions,
                input_dim=self.config.agent.input_dim,
            ).model_dump()

    def log_progress(self, ep_idx: int) -> None:
        """Displays helpful episode logs to the console during training."""
        if ep_idx % self.config.run.log_count == 0 or ep_idx == 1:
            self._metrics.log_update(self.config.run.log_count)

            print(
                f"Episode {ep_idx}/{self.config.run.episodes} | ",
                end="",
            )
            print(self._metrics)
            self._metrics = Metrics()

    def init_run(self, run_name: str) -> Analytics:
        """Creates a logging instance for analytics."""
        class_name = self.__class__.__name__
        self._analytics.init(
            project_name=f"{class_name}-{self._config.env.name}",
            run_name=run_name,
            config=ignore_empty_dicts(
                self._config.model_dump(
                    exclude_none=True,
                )
            ),
            job_type=self.env.spec.name,
            tags=[self.env.spec.name, class_name],
        )
        return self._analytics

    def train(self, run_name: str) -> None:
        """
        Trains the agent.

        Args:
            run_name (str): a unique name for the training run
            log_count (int, optional): the iteration number of episodes to log progress to the console. For example, when 100 always logs details every 100 episodes (default is 100)
        """
        run = self.init_run(run_name)
        global_step = 0
        start_time = time.time()
        num_updates = self.config.run.timesteps // self._batch_size
        state, info = torch.tensor(self.env.reset()).to(self.device)
        next_done = torch.zeros(self.config.env.n_envs).to(self.device)

        for update in range(1, num_updates + 1):
            if self.config.agent.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lr_now = frac * self.config.optimizer.lr
                self._optimizer.param_groups[0]["lr"] = lr_now

        for ep_idx in range(1, self._config.run.episodes + 1):
            terminated = False
            timesteps = 0

            for t_step in range(1, self._config.run.timesteps + 1):
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                loss = self.agent.step(state, next_state, action, reward)

                self._metrics.ep_update(reward, loss)

                state = next_state

                timesteps = t_step
                if terminated or truncated:
                    self.agent.termination()
                    break

            self._metrics.norm_ep(timesteps)

            run.log(self._metrics.ep_dict())
            self.log_progress(ep_idx)

            self._metrics.batch_update(terminated)
            self.agent.finalize_episode()

        run.finish()

    def predict(self) -> None:
        """Uses the trained agent to make a prediction."""
        pass
