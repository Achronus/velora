from typing import Any

import torch
from pydantic import ConfigDict, PrivateAttr

from velora.agent.policy import EpsilonPolicy
from velora.agent.value import QTable

from velora.config import Config
from velora.models.base import AgentModel


class SarsaBase(AgentModel):
    """
    A base class for all Sarsa agents.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        num_states (int): the discrete number of states in the environment
        num_actions (int): the discrete number of actions the agent can take
        device (torch.device): device to run computations on, such as `cpu`, `cuda`
    """

    config: Config
    num_states: int
    num_actions: int
    device: torch.device

    _vf: QTable = PrivateAttr(...)
    _policy: EpsilonPolicy = PrivateAttr(...)
    _config_exclusions: list[str] = PrivateAttr(default=["model"])

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def vf(self) -> QTable:
        return self._vf

    @property
    def policy(self) -> EpsilonPolicy:
        return self._policy

    def model_post_init(self, __context: Any) -> None:
        self._vf = QTable(
            num_states=self.num_states,
            num_actions=self.num_actions,
            device=self.device,
        )
        self._policy = EpsilonPolicy(
            device=self.device,
            **self.config.policy.model_dump(),
        )

    def td_error(self, Qsa: float, Qsa_next: float, reward: float) -> float:
        """Computes the TD error for Q-updates."""
        return reward + self.config.agent.gamma * Qsa_next - Qsa

    def q_update(self, td_error: float) -> float:
        """Performs a Q-update."""
        return self.config.agent.alpha * td_error

    def act(self, state: Any) -> int:
        return self.policy.greedy_action(self._vf[state])

    def termination(self) -> None:
        pass  # pragma: no cover

    def finalize_episode(self) -> None:
        self.policy.decay_linear()


class Sarsa(SarsaBase):
    """
    A Sarsa agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        num_states (int): the discrete number of states in the environment
        num_actions (int): the discrete number of actions the agent can take
        device (torch.device): device to run computations on, such as `cpu`, `cuda`
    """

    def step(self, state: Any, next_state: Any, action: int, reward: float) -> float:
        next_action = self.policy.greedy_action(self._vf[next_state])
        self._next_action = next_action

        td_error = self.td_error(
            self._vf[state][action],
            self._vf[next_state][next_action],
            reward,
        )
        self._vf[state][action] += self.q_update(td_error)
        return td_error


class QLearning(SarsaBase):
    """
    A Q-Learning (Sarsamax) agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        num_states (int): the discrete number of states in the environment
        num_actions (int): the discrete number of actions the agent can take
        device (torch.device): device to run computations on, such as `cpu`, `cuda`
    """

    def step(self, state: Any, next_state: Any, action: int, reward: float) -> float:
        td_error = self.td_error(
            self._vf[state][action],
            torch.max(self._vf[next_state]).item(),
            reward,
        )
        self._vf[state][action] += self.q_update(td_error)
        return td_error


class ExpectedSarsa(SarsaBase):
    """
    An Expected Sarsa agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        num_states (int): the discrete number of states in the environment
        num_actions (int): the discrete number of actions the agent can take
        device (torch.device): device to run computations on, such as `cpu`, `cuda`
    """

    def step(self, state: Any, next_state: Any, action: int, reward: float) -> float:
        action_probs = self.policy.as_dist(self._vf[state]).probs

        td_error = self.td_error(
            self._vf[state][action],
            torch.dot(self._vf[next_state], action_probs).item(),
            reward,
        )
        self._vf[state][action] += self.q_update(td_error)
        return td_error
