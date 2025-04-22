try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # pragma: no cover

from typing import TYPE_CHECKING

import gymnasium as gym
import torch

if TYPE_CHECKING:
    from velora.models.base import RLModuleAgent  # pragma: no cover

from velora.buffer.base import BufferBase
from velora.buffer.experience import BatchExperience
from velora.models.config import BufferConfig


class ReplayBuffer(BufferBase):
    """
    A Buffer for storing agent experiences. Used for Off-Policy agents.

    First introduced in Deep RL in the Deep Q-Network paper:
    [Player Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602).
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters:
            capacity (int): the total capacity of the buffer
            state_dim (int): dimension of state observations
            action_dim (int): dimension of actions
            hidden_dim (int): dimension of hidden state
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(capacity, state_dim, action_dim, hidden_dim, device=device)

    def config(self) -> BufferConfig:
        """
        Creates a buffer config model.

        Returns:
            config (BufferConfig): a config model with buffer details.
        """
        return BufferConfig(
            type="ReplayBuffer",
            capacity=self.capacity,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )

    @override
    def sample(self, batch_size: int) -> BatchExperience:
        """
        Samples a random batch of experiences from the buffer.

        Parameters:
            batch_size (int): the number of items to sample

        Returns:
            batch (BatchExperience): an object of samples with the attributes (`states`, `actions`, `rewards`, `next_states`, `dones`, `hidden`).

                All items have the same shape `(batch_size, features)`.
        """
        if len(self) < batch_size:
            raise ValueError(
                f"Buffer does not contain enough experiences. Available: {len(self)}, Requested: {batch_size}"
            )

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return BatchExperience(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
            hiddens=self.hiddens[indices],
        )

    def warm(self, agent: "RLModuleAgent", n_samples: int, num_envs: int = 8) -> None:
        """
        Warms the buffer to fill it to a number of samples by generating them
        from an agent using a `vectorized` copy of the environment.

        Parameters:
            agent (Any): the agent to generate samples with
            n_samples (int): the maximum number of samples to generate
            num_envs (int, optional): number of vectorized environments. Cannot
                be smaller than `2`
        """
        if num_envs < 2:
            raise ValueError(f"'{num_envs=}' cannot be smaller than 2.")

        envs = gym.make_vec(
            agent.env.spec.id,
            num_envs=num_envs,
            vectorization_mode="sync",
        )
        envs: gym.vector.SyncVectorEnv = gym.wrappers.vector.NumpyToTorch(
            envs, agent.device
        )

        hidden = None
        states, _ = envs.reset()

        while not len(self) >= n_samples:
            actions, hidden = agent.predict(states, hidden, train_mode=True)
            next_states, rewards, terminated, truncated, _ = envs.step(actions)
            dones = terminated | truncated

            self.add_multi(states, actions, rewards, next_states, dones, hidden)

            states = next_states

        envs.close()
