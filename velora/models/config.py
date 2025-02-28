from typing import List, Literal
from pydantic import BaseModel


class BufferConfig(BaseModel):
    """
    A config model for buffer details.

    Parameters:
        type (Literal["ReplayBuffer", "RolloutBuffer"]): the type of buffer
        capacity (int): the maximum capacity of the buffer
    """

    type: Literal["ReplayBuffer", "RolloutBuffer"]
    capacity: int


class TorchConfig(BaseModel):
    """
    A config model for PyTorch details.

    Parameters:
        device (str): the device used to train the model
        optimizer (str): the type of optimizer used
        loss (str): the type of optimizer used
    """

    device: str
    optimizer: str
    loss: str


class TrainConfig(BaseModel):
    """
    A config model for training parameter details.

    Parameters:
        batch_size (int): the size of the training batch
        n_episodes (int): the total number of episodes trained for
        max_steps (int): the maximum number of steps per training episode
        window_size (int): the episodic rate for calculating the reward moving
            average
        gamma (float): the reward discount factor
        noise_scale (int, optional): the exploration noise added when selecting
            an action (if applicable)
        tau (float, optional): the soft update factor used to slowly update the
            target networks (if applicable)
        callbacks (List[str], optional): a list of the names for callbacks used
    """

    batch_size: int
    n_episodes: int
    max_steps: int
    window_size: int
    gamma: float
    tau: float | None = None
    noise_scale: float | None = None
    callbacks: List[str] | None = None


class RLAgentConfig(BaseModel):
    """
    A config model for RL agents. Stored with agent states during the `save()` method.

    Parameters:
        agent (str): the type of agent used
        state_dim (int): number of input features
        n_neurons (int): number of hidden node
        action_dim (int): number of output features
        env (str, optional): the name of the environment the model was trained on
        model_type (Literal["actor-critic"]): the type of architecture used
        target_networks (bool): whether the agent uses target networks or not
        action_noise (Literal["OUNoise"], optional): the type of action noise
            used (if applicable)
        buffer (BufferConfig): the buffer details
        torch (TorchConfig): the PyTorch details
        train_params (TrainConfig, optional): the agents training parameters
    """

    agent: str
    state_dim: int
    n_neurons: int
    action_dim: int
    env: str | None = None
    model_type: Literal["actor-critic"]
    target_networks: bool
    action_noise: Literal["OUNoise"] | None = None
    buffer: BufferConfig
    torch: TorchConfig
    train_params: TrainConfig | None = None
