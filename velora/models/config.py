from typing import Any, Dict, Literal, Self

from pydantic import BaseModel


class BufferConfig(BaseModel):
    """
    A config model for buffer details.

    Attributes:
        type: the type of buffer
        capacity: the maximum capacity of the buffer
        state_dim: dimension of state observations
        action_dim: dimension of actions
        hidden_dim: dimension of hidden state
    """

    type: Literal["ReplayBuffer", "RolloutBuffer"]
    capacity: int
    state_dim: int
    action_dim: int
    hidden_dim: int


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
    A base config model for training parameter details.

    Attributes:
        batch_size: the size of the training batch
        n_episodes: the total number of episodes trained for
        window_size: the episodic rate for calculating the reward moving
            average
        gamma: the reward discount factor
        callbacks: a dictionary of callback details
    """

    batch_size: int
    n_episodes: int
    window_size: int
    gamma: float
    callbacks: Dict[str, Any] | None = None


class TrainConfig(TrainConfig):
    """
    A config model for episodic training parameter details.

    Attributes:
        batch_size: the size of the training batch
        n_episodes: the total number of episodes trained for
        window_size: the episodic rate for calculating the reward moving
            average
        gamma: the reward discount factor
        callbacks: a dictionary of callback details
        max_steps: the maximum number of steps per training episode
        noise_scale: the exploration noise added when selecting
            an action (if applicable)
        tau: the soft update factor used to slowly update the
            target networks (if applicable)
    """

    max_steps: int
    tau: float | None = None
    noise_scale: float | None = None


class ModuleConfig(BaseModel):
    """
    A config model for a module's details.

    Attributes:
        active_params: active module parameters count
        total_params: total module parameter count
        architecture: a summary of the module's architecture
    """

    active_params: int
    total_params: int
    architecture: Dict[str, Any]


class ModelDetails(BaseModel):
    """
    A config model for storing an agent's network model details.

    Attributes:
        type: the type of architecture used
        state_dim: number of input features
        n_neurons: number of hidden node
        action_dim: number of output features
        action_type: the type of action space
        target_networks: whether the agent uses target networks or not
        action_noise: the type of action noise used (if applicable).
            Default is `None`
        actor: details about the Actor network
        critic: details about the Critic network
    """

    type: Literal["actor-critic"]
    state_dim: int
    n_neurons: int
    action_dim: int
    action_type: Literal["discrete", "continuous"] = "continuous"
    target_networks: bool = False
    action_noise: Literal["OUNoise"] | None = None
    actor: ModuleConfig
    critic: ModuleConfig


class RLAgentConfig(BaseModel):
    """
    A config model for RL agents. Stored with agent states during the `save()` method.

    Attributes:
        agent: the type of agent used
        env: the name of the environment the model was trained on. Default is `None`
        seed: random number generator value
        model_details: the agent's network model details
        buffer: the buffer details
        torch: the PyTorch details
        train_params: the agents training parameters. Default is `None`
    """

    agent: str
    env: str | None = None
    seed: int
    model_details: ModelDetails
    buffer: BufferConfig
    torch: TorchConfig
    train_params: TrainConfig | None = None

    def update(self, env: str, train_params: TrainConfig) -> Self:
        """
        Updates the training details of the model.

        Parameters:
            env (str): the environment name
            train_params (TrainConfig): a config containing training parameters

        Returns:
            self (Self): a new config model with the updated values.
        """
        return RLAgentConfig(
            env=env,
            train_params=train_params,
            **self.model_dump(exclude={"env", "train_params"}),
        )
