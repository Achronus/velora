from typing import List, Literal

from pydantic import BaseModel


class BufferConfig(BaseModel):
    """
    A config model for buffer details.

    Attributes:
        type: the type of buffer
        capacity: the maximum capacity of the buffer
    """

    type: Literal["ReplayBuffer", "RolloutBuffer"]
    capacity: int


class TorchConfig(BaseModel):
    """
    A config model for PyTorch details.

    Attributes:
        device: the device used to train the model
        optimizer: the type of optimizer used
        loss: the type of optimizer used
    """

    device: str
    optimizer: str
    loss: str


class TrainConfig(BaseModel):
    """
    A config model for training parameter details.

    Attributes:
        batch_size: the size of the training batch
        n_episodes: the total number of episodes trained for
        max_steps: the maximum number of steps per training episode
        window_size: the episodic rate for calculating the reward moving
            average
        gamma: the reward discount factor
        noise_scale: the exploration noise added when selecting
            an action (if applicable)
        tau: the soft update factor used to slowly update the
            target networks (if applicable)
        callbacks: a list of the names for callbacks used
    """

    batch_size: int
    n_episodes: int
    max_steps: int
    window_size: int
    gamma: float
    tau: float | None = None
    noise_scale: float | None = None
    callbacks: List[str] | None = None


class ModelDetails(BaseModel):
    """
    A config model for storing an agent's network model details.

    Attributes:
        type: the type of architecture used
        state_dim: number of input features
        n_neurons: number of hidden node
        action_dim: number of output features
        target_networks: whether the agent uses target networks or not
        action_noise: the type of action noise used (if applicable).
            Default is `None`
    """

    type: Literal["actor-critic"]
    state_dim: int
    n_neurons: int
    action_dim: int
    target_networks: bool
    action_noise: Literal["OUNoise"] | None = None


class RLAgentConfig(BaseModel):
    """
    A config model for RL agents. Stored with agent states during the `save()` method.

    Attributes:
        agent: the type of agent used
        env: the name of the environment the model was trained on. Default is `None`
        model_details: the agent's network model details
        buffer: the buffer details
        torch: the PyTorch details
        train_params: the agents training parameters. Default is `None`
    """

    agent: str
    env: str | None = None
    model_details: ModelDetails
    buffer: BufferConfig
    torch: TorchConfig
    train_params: TrainConfig | None = None
