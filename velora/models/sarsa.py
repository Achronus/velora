from abc import abstractmethod

import torch
from pydantic import ConfigDict, PrivateAttr
import gymnasium as gym
from wandb.sdk.wandb_run import Run

from velora.agent.policy import EpsilonPolicy
from velora.agent.value import QTable

from velora.analytics.wandb import WeightsAndBiases

from velora.config import Config
from velora.models.base import AgentModel

from velora.utils import ignore_empty_dicts


class SarsaBase(AgentModel):
    """
    A base class for all Sarsa agents.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        env (gym.Env): the Gymnasium environment
        seed (int, optional): an random seed value for consistent experiments (default is 23)
        device (torch.device, optional): Device to run computations on, such as `cpu`, `cuda` (Default is cpu)
    """

    config: Config
    env: gym.Env
    seed: int = 23
    device: torch.device = torch.device("cpu")

    _Q: QTable = PrivateAttr(...)
    _policy: EpsilonPolicy = PrivateAttr(...)
    _analytics: WeightsAndBiases = PrivateAttr(...)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def Q(self) -> QTable:
        return self._Q

    @property
    def policy(self) -> EpsilonPolicy:
        return self._policy

    def model_post_init(self, __context):
        self._Q = QTable(
            num_states=self.env.observation_space.n,
            num_actions=self.env.action_space.n,
            device=self.device,
        )
        self._policy = EpsilonPolicy(
            device=self.device,
            **self.config.policy.model_dump(),
        )
        self._analytics = WeightsAndBiases()
        torch.manual_seed(self.seed)

    def q_update(self, Qsa: float, Qsa_next: float, reward: float) -> float:
        """Performs a Q-update."""
        return self.config.agent.alpha * (
            reward + self.config.agent.gamma * Qsa_next - Qsa
        )

    def init_run(self, run_name: str) -> Run:
        """Creates a run instance for W&B."""
        class_name = self.__class__.__name__
        return self._analytics.init(
            project_name=f"{class_name}-{self.config.env.name}",
            run_name=run_name,
            config=ignore_empty_dicts(
                self.config.model_dump(exclude="model", exclude_none=True)
            ),
            job_type=self.env.spec.name,
            tags=[self.env.spec.name, class_name],
        )

    @abstractmethod
    def train(self, run_name: str) -> QTable:
        """Trains the agents."""
        pass


class Sarsa(SarsaBase):
    """
    A Sarsa agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        env (gym.Env): the Gymnasium environment
        seed (int, optional): an random seed value for consistent experiments (default is 23)
        device (torch.device, optional): Device to run computations on, such as `cpu`, `cuda` (Default is cpu)
    """

    def train(self, run_name: str) -> QTable:
        run = self.init_run(run_name)

        for i_ep in range(1, self.config.training.episodes + 1):
            if i_ep % 100 == 0:
                print(f"Episode {i_ep}/{self.config.training.episodes}")

            score = 0
            state, _ = self.env.reset()
            action = self.policy.greedy_action(self._Q[state])

            for t_step in range(1, self.config.training.timesteps + 1):
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.policy.greedy_action(self._Q[next_state])

                self._Q[state][action] += self.q_update(
                    self._Q[state][action],
                    self._Q[next_state][next_action],
                    reward,
                )

                score += reward
                state, action = next_state, next_action

                if terminated or truncated:
                    break

            run.log({"score": score})
            self.policy.decay_linear()

        run.finish()
        return self._Q


class QLearning(SarsaBase):
    """
    A Q-Learning (Sarsamax) agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        env (gym.Env): the Gymnasium environment
        seed (int, optional): an random seed value for consistent experiments (default is 23)
        device (torch.device, optional): Device to run computations on, such as `cpu`, `cuda` (Default is cpu)
    """

    def train(self, run_name: str) -> QTable:
        run = self.init_run(run_name)

        for i_ep in range(1, self.config.training.episodes + 1):
            if i_ep % 100 == 0:
                print(f"Episode {i_ep}/{self.config.training.episodes}")

            score = 0
            state, _ = self.env.reset()

            for t_step in range(1, self.config.training.timesteps + 1):
                action = self.policy.greedy_action(self._Q[state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                self._Q[state][action] += self.q_update(
                    self._Q[state][action],
                    torch.max(self._Q[next_state]).item(),
                    reward,
                )

                score += reward
                state = next_state

                if terminated or truncated:
                    break

            run.log({"score": score})
            self.policy.decay_linear()

        run.finish()
        return self._Q


class ExpectedSarsa(SarsaBase):
    """
    An Expected Sarsa agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        env (gym.Env): the Gymnasium environment
        seed (int, optional): an random seed value for consistent experiments (default is 23)
        device (torch.device, optional): Device to run computations on, such as `cpu`, `cuda` (Default is cpu)
    """

    def train(self, run_name: str) -> QTable:
        run = self.init_run(run_name)

        for i_ep in range(1, self.config.training.episodes + 1):
            if i_ep % 100 == 0:
                print(f"Episode {i_ep}/{self.config.training.episodes}")

            score = 0
            state, _ = self.env.reset()

            for t_step in range(1, self.config.training.timesteps + 1):
                action = self.policy.greedy_action(self._Q[state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                action_probs = self.policy.soft_probs(self._Q[state]).probs

                self._Q[state][action] += self.q_update(
                    self._Q[state][action],
                    torch.dot(self._Q[next_state], action_probs).item(),
                    reward,
                )

                score += reward
                state = next_state

                if terminated or truncated:
                    break

            run.log({"score": score})
            self.policy.decay_linear()

        run.finish()
        return self._Q
