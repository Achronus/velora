import random
from typing import List, override

import gymnasium as gym
import torch

from velora.buffer.base import BufferBase
from velora.buffer.experience import BatchExperience, Experience
from velora.gym.wrap import add_core_env_wrappers
from velora.models.base import RLAgent
from velora.models.config import BufferConfig


class ReplayBuffer(BufferBase):
    """
    A Buffer for storing agent experiences. Used for Off-Policy agents.

    First introduced in Deep RL in the Deep Q-Network paper:
    [Player Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602).
    """

    def __init__(self, capacity: int, *, device: torch.device | None = None) -> None:
        """
        Parameters:
            capacity (int): the total capacity of the buffer
            device (torch.device, optional): the device to perform computations on
        """
        super().__init__(capacity, device=device)

    @property
    def config(self) -> BufferConfig:
        """
        Creates a buffer config model.

        Returns:
            config (BufferConfig): a config model with buffer details.
        """
        return BufferConfig(type="ReplayBuffer", capacity=self.capacity)

    @override
    def sample(self, batch_size: int) -> BatchExperience:
        """
        Samples a random batch of experiences from the buffer.

        Parameters:
            batch_size (int): the number of items to sample

        Returns:
            batch (BatchExperience): an object of samples with the attributes (`states`, `actions`, `rewards`, `next_states`, `dones`).

                All items have the same shape `(batch_size, features)`.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer does not contain enough experiences. Available: {len(self.buffer)}, Requested: {batch_size}"
            )

        batch: List[Experience] = random.sample(self.buffer, batch_size)
        return self._batch(batch)

    def warm(self, agent: RLAgent, env_name: str, n_samples: int) -> None:
        """
        Warms the buffer to fill it to a number of samples by generating them
        from an agent using a copy of the environment.

        Parameters:
            agent (RLAgent): the agent to generate samples with
            env_name (str): the name of environment to generate samples from
            n_samples (int): the maximum number of samples to generate
        """
        env = gym.make(env_name)
        env = add_core_env_wrappers(env, agent.device)

        print(f"\nWarming buffer with {n_samples=}...", end=" ")
        hidden = None
        state, _ = env.reset()

        while not len(self.buffer) >= n_samples:
            action, hidden = agent.predict(state, hidden)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.push(Experience(state, action.item(), reward, next_state, done))

            state = next_state

            if done:
                state, _ = env.reset()

        env.close()
        print("Complete.")
