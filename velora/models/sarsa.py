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
    _next_action: int | None = PrivateAttr(default=None)

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

    def q_update(self, Qsa: float, Qsa_next: float, reward: float) -> float:
        """Performs a Q-update."""
        return self.config.agent.alpha * (
            reward + self.config.agent.gamma * Qsa_next - Qsa
        )


class Sarsa(SarsaBase):
    """
    A Sarsa agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        num_states (int): the discrete number of states in the environment
        num_actions (int): the discrete number of actions the agent can take
        device (torch.device): device to run computations on, such as `cpu`, `cuda`
    """

    def log_progress(self, ep_idx: int, log_count: int) -> None:
        if ep_idx % log_count == 0:
            print(f"Episode {ep_idx}/{self.config.training.episodes}")

    def act(self, state: Any) -> int:
        return self.policy.greedy_action(self._vf[state])

    def step(self, state: Any, next_state: Any, action: int, reward: float) -> None:
        next_action = self.policy.greedy_action(self._vf[next_state])
        self._next_action = next_action

        self._vf[state][action] += self.q_update(
            self._vf[state][action],
            self._vf[next_state][next_action],
            reward,
        )

    def termination(self) -> None:
        pass  # pragma: no cover

    def finalize_episode(self) -> None:
        self.policy.decay_linear()


class QLearning(SarsaBase):
    """
    A Q-Learning (Sarsamax) agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        num_states (int): the discrete number of states in the environment
        num_actions (int): the discrete number of actions the agent can take
        device (torch.device): device to run computations on, such as `cpu`, `cuda`
    """

    def log_progress(self, ep_idx: int, log_count: int) -> None:
        if ep_idx % log_count == 0:
            print(f"Episode {ep_idx}/{self.config.training.episodes}")

    def act(self, state: Any) -> int:
        return self.policy.greedy_action(self._vf[state])

    def step(self, state: Any, next_state: Any, action: int, reward: float) -> None:
        self._vf[state][action] += self.q_update(
            self._vf[state][action],
            torch.max(self._vf[next_state]).item(),
            reward,
        )

    def termination(self) -> None:
        pass  # pragma: no cover

    def finalize_episode(self) -> None:
        self.policy.decay_linear()


class ExpectedSarsa(SarsaBase):
    """
    An Expected Sarsa agent.

    Args:
        config (velora.Config): a Config model loaded from a YAML file
        num_states (int): the discrete number of states in the environment
        num_actions (int): the discrete number of actions the agent can take
        device (torch.device): device to run computations on, such as `cpu`, `cuda`
    """

    def log_progress(self, ep_idx: int, log_count: int) -> None:
        if ep_idx % log_count == 0:
            print(f"Episode {ep_idx}/{self.config.training.episodes}")

    def act(self, state: Any) -> int:
        return self.policy.greedy_action(self._vf[state])

    def step(self, state: Any, next_state: Any, action: int, reward: float) -> None:
        action_probs = self.policy.as_dist(self._vf[state]).probs

        self._vf[state][action] += self.q_update(
            self._vf[state][action],
            torch.dot(self._vf[next_state], action_probs).item(),
            reward,
        )

    def termination(self) -> None:
        pass  # pragma: no cover

    def finalize_episode(self) -> None:
        self.policy.decay_linear()
